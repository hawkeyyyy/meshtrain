import pytest
import torch

from meshtrain.worker.device import (
    CPUDeviceAdapter,
    CUDADeviceAdapter,
    MPSDeviceAdapter,
    available_backends,
    select_device,
)


def test_cpu_adapter_basics():
    a = CPUDeviceAdapter()
    assert a.backend == "cpu" and a.device == torch.device("cpu")
    t = a.move_tensor(torch.ones(2))
    a.synchronize()
    stats = a.memory_stats()
    assert stats["total"] > 0 and not stats["unified"]
    assert a.supports("float64") and not a.supports("nccl")
    assert t.device.type == "cpu"


def test_select_device_fallback_and_explicit():
    assert available_backends()[-1] == "cpu"
    assert select_device("cpu").backend == "cpu"
    assert select_device("auto").backend == available_backends()[0]
    with pytest.raises(ValueError):
        select_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_cuda_adapter_refuses_without_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        CUDADeviceAdapter()


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS present")
def test_mps_adapter_refuses_without_mps():
    with pytest.raises(RuntimeError, match="MPS"):
        MPSDeviceAdapter()


def test_mps_capabilities_declared_without_hardware():
    # Capability logic is static: MPS has no float64 kernels.
    a = object.__new__(MPSDeviceAdapter)
    assert not a.supports_dtype(torch.float64)
    assert a.supports_dtype(torch.float32)


@pytest.mark.cuda
def test_cuda_adapter_on_hardware():
    a = CUDADeviceAdapter()
    t = a.move_tensor(torch.randn(4))
    a.synchronize()
    assert t.is_cuda and a.memory_stats()["total"] > 0


@pytest.mark.mps
def test_mps_adapter_on_hardware():
    a = MPSDeviceAdapter()
    t = a.move_tensor(torch.randn(4))
    a.synchronize()
    assert t.device.type == "mps" and a.memory_stats()["unified"]
    with pytest.raises(TypeError):
        a.move_tensor(torch.randn(2, dtype=torch.float64))


def test_cpu_staging_is_zero_copy_and_pooled():
    a = CPUDeviceAdapter()
    t = torch.randn(4, 4)
    cpu, release = a.begin_d2h(t)()
    assert cpu.data_ptr() == t.data_ptr() and release is None
    buf = a.recv_buffer((4, 4), torch.float32)
    handle, token = a.begin_h2d(buf)
    assert a.finish_h2d(handle) is buf
    a.after_use(token)
    assert a.recv_buffer((4, 4), torch.float32) is buf  # reused after backward


@pytest.mark.cuda
def test_cuda_pinned_d2h_and_h2d_round_trip():
    a = CUDADeviceAdapter()
    a.configure_transfers(pinned=True, pool=True)
    x = torch.randn(256, 256, device="cuda")
    y = x * 2  # queued on the compute stream; begin_d2h must order after it
    cpu, release = a.begin_d2h(y)()
    assert cpu.is_pinned() and torch.allclose(cpu, (x * 2).cpu())
    release()
    buf = a.recv_buffer((256, 256), torch.float32)
    buf.copy_(cpu)
    handle, token = a.begin_h2d(buf)
    back = a.finish_h2d(handle)
    assert back.is_cuda and torch.allclose(back, x * 2)
    a.after_use(token)
    assert a.host_pool.stats()["allocation_count"] >= 1

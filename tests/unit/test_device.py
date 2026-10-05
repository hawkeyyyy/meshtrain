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

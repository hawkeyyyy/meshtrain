import pytest
import torch

from meshtrain.planner.graph import profile_model
from meshtrain.models import MLPSpec, TinyTransformerSpec
from meshtrain.planner.partition import PlannerOptions, WorkerProfile, make_plan
from meshtrain.worker.device import CPUDeviceAdapter, MPSDeviceAdapter

GB = 1024**3


def test_cpu_capability_report_is_probed():
    caps = CPUDeviceAdapter().capabilities()
    assert caps["backend"] == "cpu" and caps["supports_fp32"] and not caps["pinned_memory"]
    assert {"matmul", "sdpa", "layer_norm", "adamw"} <= set(caps["op_capabilities"]["float32"])
    assert caps["total_memory"] > 0 and caps["available_memory"] > 0


def test_supports_op_and_dtype():
    a = CPUDeviceAdapter()
    assert a.supports("matmul", "float32") and a.supports("sdpa") and a.supports("float64")
    assert not a.supports("nccl") and a.supports("some_future_op")


def test_failed_probe_is_recorded_not_raised():
    a = CPUDeviceAdapter()
    a.prober._cache.clear()
    from meshtrain.worker import capabilities as caps

    orig = caps.PROBES["sdpa"]
    caps.PROBES["sdpa"] = lambda d, t: (_ for _ in ()).throw(RuntimeError("no kernel"))
    try:
        assert not a.supports("sdpa", "float32")
        assert "sdpa:float32" in a.prober.errors
    finally:
        caps.PROBES["sdpa"] = orig


def test_mps_static_rules_without_hardware():
    a = object.__new__(MPSDeviceAdapter)
    assert not a.supports_dtype(torch.float64)


def test_planner_excludes_workers_missing_required_ops():
    layers = profile_model(TinyTransformerSpec(layers=2, hidden_size=64, heads=4, vocab_size=64, seq_len=16), 2)
    full = {"float32": ["embedding", "linear", "layer_norm", "gelu", "sdpa", "cross_entropy", "adamw"]}
    no_sdpa = {"float32": ["embedding", "linear", "layer_norm", "gelu", "cross_entropy", "adamw"]}
    ws = [WorkerProfile("a", "a", "cuda", 8 * GB, 1e12, supported_ops=full),
          WorkerProfile("m", "m", "mps", 8 * GB, 1e12, supported_ops=no_sdpa),
          WorkerProfile("u", "u", "cpu", 8 * GB, 1e11)]  # unknown capabilities: assumed fine
    plan = make_plan("auto", layers, ws, opts=PlannerOptions(
        required_ops=TinyTransformerSpec.required_ops + ("adamw",), optimizer="adamw"))
    assert "m" in plan.excluded and "sdpa" in plan.excluded["m"]
    assert "u" not in plan.excluded


def test_mlp_does_not_need_attention():
    assert "sdpa" not in MLPSpec.required_ops and "sdpa" in TinyTransformerSpec.required_ops

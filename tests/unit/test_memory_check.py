import pytest
import torch

from meshtrain.config import parse_config
from meshtrain.models import MLPSpec
from meshtrain.runtime.memory_check import StageMemoryError, format_validation, is_oom, validate_stage
from meshtrain.runtime.stage import Stage
from meshtrain.worker.device import CPUDeviceAdapter

GB = 1024**3


def _stage():
    spec = MLPSpec(sizes=[64, 512, 512, 10])
    return Stage(spec.build_stage(0, 3), stage_index=0, num_stages=2, optimizer="adamw")


def test_validation_report_and_estimation_error(monkeypatch):
    monkeypatch.delenv("MESHTRAIN_EMULATE_DEVICE_MEMORY_GB", raising=False)
    st = _stage()
    P = st.memory_report()["parameters"]
    est = {"required": 5 * P, "parameters": int(P * 0.9)}
    rep = validate_stage(st, CPUDeviceAdapter(), est, probe=False)
    assert rep["actual_parameters"] == P and rep["estimated_remaining"] == 5 * P - int(P * 0.9)
    assert rep["parameter_estimate_error"] == pytest.approx(P / int(P * 0.9) - 1)
    assert "estimated =" in format_validation(rep)


def test_dangerous_placement_fails_before_training(monkeypatch):
    st = _stage()
    P = st.memory_report()["parameters"]
    monkeypatch.setenv("MESHTRAIN_EMULATE_DEVICE_MEMORY_GB", str(2 * P / GB))  # room for ~1 more P
    with pytest.raises(StageMemoryError, match="needs") as ei:
        validate_stage(st, CPUDeviceAdapter(), {"required": 5 * P, "parameters": P}, probe=False)
    assert ei.value.report["capacity"] == pytest.approx(2 * P, rel=0.01)
    assert is_oom(ei.value)


def test_is_oom_classification():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 26.00 MiB"))
    assert is_oom(RuntimeError("MPS backend out of memory"))
    assert not is_oom(ValueError("shape mismatch"))


def test_config_round_trip_keeps_v15_memory_model():
    """Regression: a dumped + re-parsed config must not switch to the deprecated V1 headroom."""
    cfg = parse_config({"model": {"type": "mlp"}})
    again = parse_config(cfg.model_dump(mode="json", by_alias=True))
    assert again.placement.memory_headroom_fraction is None and again.placement.memory_headroom_min_gb is None
    assert again.transport.async_ is True and again.memory.safety_factors()["mps"] == 0.80

"""V2.4 residency planner, offload-aware cluster planning, timeline memory metrics, CLI."""

import pytest

from meshtrain.cli import build_parser
from meshtrain.config import parse_config
from meshtrain.coordinator.scheduler import plan_job
from meshtrain.coordinator.state import WorkerRecord
from meshtrain.models import TinyTransformerSpec
from meshtrain.planner.residency import plan_stage_residency, planned_tensor_records
from meshtrain.runtime.offload import ResidencyPolicy
from meshtrain.runtime.tensor_store import MemoryTier, TensorRole
from meshtrain.runtime.timeline import Timeline

GB = 1024**3
SPEC = TinyTransformerSpec(layers=12, hidden_size=512, heads=8, vocab_size=1024, seq_len=64)
KW = dict(microbatch_size=2, optimizer="adamw", backend="cuda", num_microbatches=4)


def _plan(policy, budget):
    return plan_stage_residency(SPEC, 0, SPEC.num_layers, budget=budget, policy=policy, **KW)


def test_static_requirement_and_infeasibility_reported():
    static = _plan(ResidencyPolicy(), 256 * 1024**2)
    assert not static.feasible and "every layer resident" in static.reason
    assert static.static_required > 256 * 1024**2
    with pytest.raises(MemoryError, match="no residency plan fits"):
        static.apply(ResidencyPolicy())


def test_auto_offload_fills_budget_with_hot_layers_and_stays_within_it():
    pol = ResidencyPolicy(strategy="auto_offload", optimizer_execution="cpu_offload")
    unlimited = _plan(pol, None)
    assert len(unlimited.hot_groups) == len(unlimited.groups)        # no budget: everything resident
    tight = _plan(pol, 300 * 1024**2)
    assert tight.feasible and 0 < len(tight.hot_groups) < len(tight.groups)
    assert tight.device_total <= 300 * 1024**2 * 0.90
    looser = _plan(pol, 600 * 1024**2)
    assert len(looser.hot_groups) > len(tight.hot_groups)             # more budget -> more resident layers
    assert looser.estimated_transfer_s < tight.estimated_transfer_s
    assert _plan(pol, 300 * 1024**2).apply(pol).resident_groups == tight.hot_groups


def test_manual_keep_resident_and_cpu_optimizer_device_bytes():
    pol = ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0", "13"))
    plan = _plan(pol, None)
    assert set(plan.hot_groups) == {"model.layers.0", "model.layers.13"}
    accel = plan.device_bytes["cold optimizer state"]
    cpu = _plan(ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0", "13"),
                                optimizer_execution="cpu_offload"), None).device_bytes
    assert accel > 0 and cpu["cold optimizer state"] == 0 and cpu["hot optimizer state"] == 0


def test_planned_tensor_records_tiers():
    pol = ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0",), optimizer_execution="cpu_offload")
    recs = planned_tensor_records(SPEC, 0, 3, _plan(pol, None), optimizer="adamw")
    by = {r.tensor_id: r for r in recs}
    assert by["model.layers.1.qkv.weight"].tier == MemoryTier.LOCAL_RAM and not by["model.layers.1.qkv.weight"].cached
    hot = by["model.layers.0.tok.weight"]
    assert hot.tier == MemoryTier.LOCAL_RAM and hot.cached == {MemoryTier.LOCAL_ACCELERATOR}  # master in RAM + copy
    assert by["model.layers.1.qkv.weight.optim.exp_avg"].role == TensorRole.OPTIMIZER_STATE
    static = planned_tensor_records(SPEC, 0, 3, None, optimizer="adamw")
    assert {r.tier for r in static} == {MemoryTier.LOCAL_ACCELERATOR}


def test_cluster_planner_offload_makes_static_infeasible_model_feasible():
    workers = [WorkerRecord("gpu", "gpu", {}, "cuda", {"memory_total": 8 * GB}, "h", 29500)]
    base = {"model": {"type": "tiny_transformer", "layers": 20, "hidden_size": 1280, "heads": 8, "vocab_size": 1024,
                      "seq_len": 64},
            "training": {"batch_size": 8, "microbatch_size": 2, "optimizer": "adamw"}, "placement": {"num_stages": 1}}
    static = plan_job(parse_config(base), workers)
    assert not static.feasible
    off = plan_job(parse_config({**base, "memory": {"strategy": "auto_offload", "optimizer_offload": True}}), workers)
    assert off.feasible and off.stages[0].memory.residency["feasible"]
    capped = plan_job(parse_config({**base, "model": {**base["model"], "layers": 4},
                                    "memory": {"accelerator_budget_mb": 512}}), workers)
    assert not capped.feasible   # static under an explicit budget is checked against the budget


def test_timeline_separates_memory_transfers_from_compute_and_network():
    tl = Timeline()
    tl.add("FORWARD_COMPUTE", 0.0, 1.0, 0)
    tl.add("TENSOR_LOAD", 0.2, 0.4, 0)            # blocking load inside the forward span
    tl.add("TENSOR_PREFETCH", 0.5, 0.9, 0)        # async, fully overlapped with compute
    tl.add("NETWORK_SEND", 1.0, 1.5, 0)
    m = tl.step_metrics(0, (0.0, 2.0))
    assert m["compute_s"] == pytest.approx(0.8)   # 1.0 minus the blocking load
    assert m["communication_s"] == pytest.approx(0.5)
    assert m["exposed_memory_transfer_s"] == pytest.approx(0.2)
    assert m["memory_overlapped_s"] == pytest.approx(0.4)
    assert m["memory_transfer_s"] == pytest.approx(0.6)


def test_v2_cli_commands_parse():
    p = build_parser()
    assert p.parse_args(["tensors", "list", "c.yaml", "--layers", "0:4"]).func.__name__ == "cmd_tensors_list"
    assert p.parse_args(["tensors", "inspect", "c.yaml", "model.layers.1.qkv.weight"]).tensor_id.endswith("weight")
    assert p.parse_args(["inspect", "tensors", "c.yaml"]).func.__name__ == "cmd_tensors_list"
    assert p.parse_args(["memory", "plan", "c.yaml", "--budget-mb", "2048"]).budget_mb == 2048
    assert p.parse_args(["memory", "status", "--run", "runs/x"]).run == "runs/x"
    b = p.parse_args(["benchmark", "offload", "--budget-mb", "2048", "--sweep", "prefetch"])
    assert b.what == "offload" and b.sweep == "prefetch"
    e = p.parse_args(["experiment", "capacity", "--memory-strategy", "static", "--memory-strategy", "auto-offload"])
    assert e.memory_strategy == ["static", "auto-offload"]
    assert p.parse_args(["experiment", "offload-correctness"]).name == "offload-correctness"

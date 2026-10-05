import pytest

from meshtrain.models import MLPSpec, TinyTransformerSpec
from meshtrain.planner.cost import NetworkModel, pipeline_step_time
from meshtrain.planner.graph import profile_model
from meshtrain.planner.memory import estimate_stage_memory
from meshtrain.planner.partition import PlannerOptions, WorkerProfile, make_plan, plan_manual

GB = 1024**3


def workers():
    return [
        WorkerProfile("a", "rtx8", "cuda", 8 * GB, 10e12),
        WorkerProfile("b", "rtx12", "cuda", 12 * GB, 14e12),
        WorkerProfile("c", "m2", "mps", 10 * GB, 3e12, unified_memory=True,
                      supported_dtypes=("float32", "float16")),
        WorkerProfile("d", "old-pc", "cpu", 4 * GB, 0.1e12),
    ]


@pytest.fixture(scope="module")
def big_layers():
    # ~0.93B params: with Adam (16 bytes/param) too big for any single device below
    return profile_model(TinyTransformerSpec(layers=32, hidden_size=1536, heads=16, vocab_size=8192, seq_len=256), 2)


def test_profile_counts_parameters_exactly():
    spec = MLPSpec()
    layers = profile_model(spec, 8)
    assert sum(l.param_count for l in layers) == spec.parameter_count() == sum(
        p.numel() for p in spec.build_full().parameters())
    assert layers[-1].output_shape == (8, 10)


def test_transfer_time_formula_and_asymmetry():
    nm = NetworkModel(latency={("a", "b"): 0.001, ("b", "a"): 0.002},
                      bandwidth={("a", "b"): 100e6, ("b", "a"): 50e6})
    assert nm.transfer_time(100e6, "a", "b") == pytest.approx(1.001)
    assert nm.transfer_time(100e6, "b", "a") == pytest.approx(2.002)
    assert pipeline_step_time([1.0, 2.0, 1.5], 4) == pytest.approx(6 * 2.0)


def test_memory_accounting_components(big_layers):
    m = estimate_stage_memory(big_layers, total=12 * GB, optimizer="adam", num_microbatches=4, is_last=False)
    assert m.gradients == m.parameters and m.optimizer_state == 2 * m.parameters
    assert m.saved_activations > 0 and m.reserved >= int(0.15 * 12 * GB)
    assert "parameters" in m.format("rtx12")


@pytest.mark.parametrize("strategy", ["equal", "compute", "auto"])
def test_plans_cover_all_layers_contiguously(strategy, big_layers):
    opts = PlannerOptions(optimizer="adam", num_microbatches=4, allow_backends=("cuda", "mps"))
    plan = make_plan(strategy, big_layers, workers(), opts=opts)
    assert plan.stages[0].start == 0 and plan.stages[-1].end == len(big_layers)
    for a, b in zip(plan.stages, plan.stages[1:]):
        assert a.end == b.start
    assert "old-pc" in plan.excluded


def test_auto_respects_memory_budgets(big_layers):
    opts = PlannerOptions(optimizer="adam", num_microbatches=4, allow_backends=("cuda", "mps"))
    plan = make_plan("auto", big_layers, workers(), opts=opts)
    assert plan.feasible, plan.format()
    for s in plan.stages:
        assert s.memory.required <= s.memory.budget
    # model needs more than any single device can hold
    single = [make_plan("auto", big_layers, [w], opts=opts) for w in workers()[:3]]
    assert not any(p.feasible for p in single)


def test_auto_beats_or_matches_equal_split(big_layers):
    opts = PlannerOptions(optimizer="adam", num_microbatches=4, allow_backends=("cuda", "mps"))
    auto = make_plan("auto", big_layers, workers(), opts=opts)
    eq = make_plan("equal", big_layers, workers(), opts=opts)
    if eq.feasible:
        assert auto.predicted_step_s <= eq.predicted_step_s * 1.001


def test_infeasible_reported_not_overcommitted(big_layers):
    tiny = [WorkerProfile("x", "tiny", "cuda", 2 * GB, 1e12), WorkerProfile("y", "tiny2", "cuda", 2 * GB, 1e12)]
    plan = make_plan("auto", big_layers, tiny, opts=PlannerOptions(optimizer="adam"))
    assert not plan.feasible and plan.violations
    eq = make_plan("equal", big_layers, tiny, opts=PlannerOptions(optimizer="adam"))
    assert not eq.feasible and any("needs" in v for v in eq.violations)


def test_dtype_support_excludes_worker():
    layers = profile_model(MLPSpec(), 8)
    plan = make_plan("auto", layers, workers(), opts=PlannerOptions(dtype="bfloat16"))
    assert "m2" in plan.excluded


def test_manual_plan():
    layers = profile_model(MLPSpec(), 8)
    plan = plan_manual(layers, workers(), NetworkModel(), PlannerOptions(), [("a", 0, 3), ("rtx12", 3, 7)])
    assert plan.feasible and [s.worker_name for s in plan.stages] == ["rtx8", "rtx12"]

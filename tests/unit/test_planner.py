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
    return profile_model(TinyTransformerSpec(layers=32, hidden_size=1536, heads=16, vocab_size=8192, seq_len=64), 2)


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


def test_regression_real_hardware_capacity_outcomes():
    """User-reported run: RTX 4070 Laptop (8 GB) + GTX 1050 Ti (3.94 GB), AdamW, batch 8 / microbatch 2.
    396M parameters trained across both; 475M passed the V1 planner and then hit CUDA OOM on the 1050 Ti.
    The V1.5 estimator must accept the first and reject the second."""
    from meshtrain.experiments.capacity import model_cfg
    from meshtrain.models import build_model_spec

    ws = [WorkerProfile("hawkey", "hawkey", "cuda", int(8.0 * GB), 10e12),
          WorkerProfile("server", "server", "cuda", int(3.94 * GB), 2e12)]
    for (blocks, hidden), should_fit in (((20, 1280), True), ((24, 1280), False)):
        layers = profile_model(build_model_spec(model_cfg(blocks, hidden)), 2)
        for schedule in ("gpipe", "1f1b"):
            plan = make_plan("auto", layers, ws, opts=PlannerOptions(optimizer="adamw", num_microbatches=4,
                                                                       schedule=schedule))
            assert plan.feasible == should_fit, (blocks, hidden, schedule)


def test_memory_estimate_components():
    layers = profile_model(TinyTransformerSpec(layers=4, hidden_size=256, heads=4, vocab_size=512, seq_len=64), 4)
    adam = estimate_stage_memory(layers, total=8 * GB, optimizer="adamw", num_microbatches=8, is_last=False,
                                 stage_index=0, num_stages=3, schedule="gpipe", backend="cuda")
    sgd = estimate_stage_memory(layers, total=8 * GB, optimizer="sgd", num_microbatches=8, is_last=False,
                                stage_index=0, num_stages=3, schedule="gpipe", backend="cuda")
    f1b = estimate_stage_memory(layers, total=8 * GB, optimizer="adamw", num_microbatches=8, is_last=False,
                                stage_index=0, num_stages=3, schedule="1f1b", backend="cuda")
    P = adam.parameters
    assert adam.optimizer_state == 2 * P and adam.optimizer_step_temporary == P and adam.gradients == P
    assert sgd.optimizer_state == 0 and sgd.optimizer_step_temporary == 0
    assert adam.in_flight == 8 and f1b.in_flight == 3
    assert f1b.saved_activations * 8 == adam.saved_activations * 3
    assert adam.usable == int(8 * GB * 0.85) - int(0.4 * GB)
    mps = estimate_stage_memory(layers, total=8 * GB, optimizer="adamw", num_microbatches=8, is_last=True,
                                backend="mps")
    assert mps.usable == int(8 * GB * 0.80) and mps.in_flight == 1

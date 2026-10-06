"""V2 integration: offloaded training must equal static (V1.5) training.

CPU runs are bit-exact (same kernels, values only copied). CUDA runs are
bit-exact for the accelerator optimizer; the CPU optimizer (cpu_offload)
differs from CUDA AdamW only by kernel rounding, which Adam amplifies on
parameters whose true gradient is ~0 (sqrt(v) << eps, e.g. attention key
biases) -- checked separately below.
"""

import multiprocessing as mp

import pytest
import torch

from meshtrain.experiments.offload_correctness import (
    DEFAULT_MODEL,
    build_single_stage,
    compare_traces,
    format_rows,
    train_trace,
)
from meshtrain.runtime.checkpoint import load_stage_checkpoint, save_stage_checkpoint
from meshtrain.runtime.local import LocalStage, global_param_names, run_local_pipeline
from meshtrain.runtime.offload import ResidencyPolicy
from meshtrain.runtime.pipeline import PipelineSettings, run_stage

STEPS = 4
POLICIES = {
    "manual sync": ResidencyPolicy(strategy="manual_offload", prefetch_distance=0),
    "manual prefetch1": ResidencyPolicy(strategy="manual_offload", prefetch_distance=1),
    "manual prefetch2": ResidencyPolicy(strategy="manual_offload", prefetch_distance=2),
    "manual after_backward": ResidencyPolicy(strategy="manual_offload", eviction="after_backward"),
    "manual keep 0,5": ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0", "layers.5")),
    "optimizer offload": ResidencyPolicy(strategy="static", optimizer_execution="cpu_offload"),
    "manual + optimizer offload": ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"),
}


@pytest.fixture(scope="module")
def cpu_baseline():
    return train_trace(DEFAULT_MODEL, steps=STEPS)


@pytest.mark.parametrize("name", list(POLICIES))
def test_cpu_offload_is_bit_exact_every_step(cpu_baseline, name):
    tr = train_trace(DEFAULT_MODEL, policy=POLICIES[name], steps=STEPS)
    rows = compare_traces(cpu_baseline, tr)
    print(f"\n{name}\n{format_rows(rows)}")
    for r in rows:
        assert r["loss_abs_diff"] == 0 and r["grad_max_abs_diff"] == 0, r
        assert r["param_max_abs_diff"] == 0 and r["logits_max_abs_diff"] == 0, r
    assert tr.stats[-1]["residency"]["tensor_eviction_count"] > 0 or name == "optimizer offload"


def test_sgd_momentum_free_and_adam_paths(cpu_baseline):
    base = train_trace(DEFAULT_MODEL, steps=3, optimizer="sgd", lr=0.05)
    off = train_trace(DEFAULT_MODEL, steps=3, optimizer="sgd", lr=0.05,
                      policy=ResidencyPolicy(strategy="manual_offload"))
    assert all(r["param_max_abs_diff"] == 0 for r in compare_traces(base, off))


def test_low_budget_auto_offload_trains_within_budget(cpu_baseline):
    from meshtrain.models import build_model_spec
    from meshtrain.planner.residency import plan_stage_residency

    model = {**DEFAULT_MODEL, "layers": 6}
    spec = build_model_spec(model)
    pol = ResidencyPolicy(strategy="auto_offload", optimizer_execution="cpu_offload", prefetch_distance=1)
    kw = dict(microbatch_size=2, optimizer="adamw", backend="cpu")
    static = plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=ResidencyPolicy(), **kw)
    minimal = plan_stage_residency(spec, 0, spec.num_layers, budget=None,
                                   policy=ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"),
                                   **kw)
    all_hot = plan_stage_residency(spec, 0, spec.num_layers, budget=None,
                                   policy=ResidencyPolicy(optimizer_execution="cpu_offload"), **kw)
    budget = int((minimal.device_total + all_hot.device_total) / 2 / 0.90)
    # static (V1.5) does not fit; neither does "everything resident"; offloading some layers does
    assert minimal.device_total < budget * 0.90 < all_hot.device_total < static.device_total
    plan = plan_stage_residency(spec, 0, spec.num_layers, budget=budget, policy=pol, **kw)
    assert plan.feasible and 0 < len(plan.hot_groups) < len(plan.groups)
    base = train_trace(model, steps=3)
    tr = train_trace(model, steps=3, policy=pol, budget_bytes=budget)
    for st in tr.stats:
        r = st["residency"]
        assert r["accelerator_resident_peak"] <= budget and r["budget_violations"] == 0
        assert r["groups_cold"] == len(plan.groups) - len(plan.hot_groups)
    assert all(r["param_max_abs_diff"] == 0 for r in compare_traces(base, tr))


def test_two_process_pipeline_with_offload_matches_static():
    stages = [LocalStage((0, 3)), LocalStage((3, 6))]
    settings = PipelineSettings("v2-pipe", steps=3, batch_size=8, num_microbatches=4, capture_gradients_at_step=2,
                                schedule="1f1b", async_transport=True)
    kw = dict(seed=0, optimizer="adamw", lr=1e-3, capture_params=True, transport="tcp")
    static = run_local_pipeline(DEFAULT_MODEL, stages, settings, **kw)
    off = run_local_pipeline(DEFAULT_MODEL, stages, settings, **kw,
                             residency=ResidencyPolicy(strategy="manual_offload", prefetch_distance=1))
    assert static[0].losses == off[0].losses
    for a, b in zip(static, off):
        ga = global_param_names(stages, a.gradients, a.stage_index)
        gb = global_param_names(stages, b.gradients, b.stage_index)
        assert all(torch.equal(ga[k], gb[k]) for k in ga)
        assert all(torch.equal(a.final_params[k], b.final_params[k]) for k in a.final_params)
        assert "residency" in b.step_metrics[-1] and b.step_metrics[-1]["residency"]["tensor_eviction_count"] > 0


def _run(stage, spec, steps, offset):
    for i in range(offset, offset + steps):
        s = PipelineSettings("ckpt", steps=1, batch_size=8, num_microbatches=4, step_offset=i, trace=False,
                             log_every=10**6)
        run_stage(stage, spec, s, upstream=None, downstream=None)


@pytest.mark.parametrize("save_policy,load_policy", [
    (ResidencyPolicy(strategy="manual_offload"), None),
    (None, ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload")),
    (ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"),
     ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.1",))),
])
def test_checkpoint_round_trip_is_residency_independent(tmp_path, save_policy, load_policy):
    ref, spec = build_single_stage(DEFAULT_MODEL)
    _run(ref, spec, 4, 0)
    want = ref.named_parameters_cpu()
    a, spec = build_single_stage(DEFAULT_MODEL, policy=save_policy)
    _run(a, spec, 2, 0)
    save_stage_checkpoint(a, tmp_path / "s.pt", step=2)
    a.close()
    b, spec = build_single_stage(DEFAULT_MODEL, policy=load_policy)
    info = load_stage_checkpoint(b, tmp_path / "s.pt")
    assert info["step"] == 2
    _run(b, spec, 2, 2)
    got = b.named_parameters_cpu()
    assert all(torch.equal(want[k], got[k]) for k in want)   # resumed == uninterrupted, bit-exact on CPU
    b.close()


# -- CUDA --------------------------------------------------------------------------
@pytest.mark.cuda
@pytest.mark.parametrize("name", ["manual sync", "manual prefetch1", "manual after_backward", "manual keep 0,5"])
def test_cuda_parameter_offload_is_bit_exact(name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    base = train_trace(DEFAULT_MODEL, device="cuda", steps=STEPS)
    tr = train_trace(DEFAULT_MODEL, device="cuda", policy=POLICIES[name], steps=STEPS)
    rows = compare_traces(base, tr)
    print(f"\n{name}\n{format_rows(rows)}")
    assert all(r["param_max_abs_diff"] == 0 and r["loss_abs_diff"] == 0 and r["grad_max_abs_diff"] == 0
               for r in rows)


@pytest.mark.cuda
def test_cuda_cpu_optimizer_offload_matches_within_adam_rounding():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pol = POLICIES["manual + optimizer offload"]
    base = train_trace(DEFAULT_MODEL, device="cuda", steps=STEPS)
    tr = train_trace(DEFAULT_MODEL, device="cuda", policy=pol, steps=STEPS)
    rows = compare_traces(base, tr)
    print("\n" + format_rows(rows))
    for r in rows:
        assert r["loss_abs_diff"] < 1e-5 and r["grad_max_abs_diff"] < 1e-6
        assert r["param_mean_abs_diff"] < 1e-7 and r["param_max_abs_diff"] < 1e-3 * 0.2  # < 20% of one lr step
        assert r["logits_max_abs_diff"] < 1e-4


def _cap_child(q, strategy, cap_mb):
    import os

    os.environ["MESHTRAIN_QUIET"] = "1"
    from meshtrain.experiments.offload_correctness import train_trace as tt
    from meshtrain.runtime.offload import ResidencyPolicy as RP

    model = {"type": "tiny_transformer", "layers": 8, "hidden_size": 512, "heads": 8, "vocab_size": 1024,
             "seq_len": 64}
    pol = RP(strategy=strategy, optimizer_execution="cpu_offload" if strategy != "static" else "accelerator")
    try:
        tr = tt(model, device="cuda", policy=pol, budget_bytes=cap_mb * 1024**2, steps=2, batch_size=4,
                microbatch_size=2, capture=False)
        peak = torch.cuda.max_memory_allocated()
        q.put(("ok", tr.losses, peak))
    except Exception as exc:
        q.put(("fail", f"{type(exc).__name__}: {str(exc)[:200]}", None))


@pytest.mark.cuda
def test_cuda_static_ooms_but_offload_trains_under_same_cap():
    """Real allocator cap: ~30M-parameter model, 256 MB cap. Static needs ~0.5 GB of model state."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ctx = mp.get_context("spawn")
    out = {}
    for strategy in ("static", "auto_offload"):
        q = ctx.Queue()
        p = ctx.Process(target=_cap_child, args=(q, strategy, 256))
        p.start()
        out[strategy] = q.get(timeout=600)
        p.join(30)
    assert out["static"][0] == "fail" and "out of memory" in out["static"][1].lower(), out["static"]
    assert out["auto_offload"][0] == "ok", out["auto_offload"]
    assert out["auto_offload"][2] <= 256 * 1024**2


def _optimizer_state(stage) -> dict:
    out = {}
    for n, p in stage.module.named_parameters():
        for k, v in stage.optimizer.state.get(p, {}).items():
            if torch.is_tensor(v):
                out[f"{n}.{k}"] = v.detach().float().cpu().clone()
    return out


def _losses(stage, spec, steps, offset):
    out = []
    for i in range(offset, offset + steps):
        s = PipelineSettings("ckpt", steps=1, batch_size=8, num_microbatches=4, step_offset=i, trace=False,
                             log_every=10**6)
        out.append(run_stage(stage, spec, s, upstream=None, downstream=None).losses[0])
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("save_policy,load_policy,exact", [
    (None, ResidencyPolicy(strategy="manual_offload"), True),
    (ResidencyPolicy(strategy="manual_offload", prefetch_distance=1), None, True),
    (None, ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"), False),
    (ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"), None, False),
])
def test_cuda_checkpoint_resume_across_residency(tmp_path, save_policy, load_policy, exact):
    """static CUDA <-> local-offload CUDA: loss, parameters and AdamW state after resuming."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ref, spec = build_single_stage(DEFAULT_MODEL, device="cuda")
    ref_losses = _losses(ref, spec, 4, 0)
    want_p, want_s = ref.named_parameters_cpu(), _optimizer_state(ref)
    ref.close()
    a, spec = build_single_stage(DEFAULT_MODEL, device="cuda", policy=save_policy)
    first = _losses(a, spec, 2, 0)
    save_stage_checkpoint(a, tmp_path / "s.pt", step=2)
    a.close()
    b, spec = build_single_stage(DEFAULT_MODEL, device="cuda", policy=load_policy)
    load_stage_checkpoint(b, tmp_path / "s.pt")
    second = _losses(b, spec, 2, 2)
    got_p, got_s = b.named_parameters_cpu(), _optimizer_state(b)
    b.close()
    assert set(got_s) == set(want_s) and set(got_p) == set(want_p)
    if exact:   # same GPU AdamW kernels on both sides of the checkpoint
        assert first + second == ref_losses
        assert all(torch.equal(want_p[k], got_p[k]) for k in want_p)
        assert all(torch.equal(want_s[k], got_s[k]) for k in want_s)
    else:       # one half ran CPU AdamW: kernel rounding only (see module docstring)
        assert first + second == pytest.approx(ref_losses, abs=1e-5)
        assert max(float((want_p[k] - got_p[k]).abs().max()) for k in want_p) < 2e-4
        assert max(float((want_s[k] - got_s[k]).abs().max()) for k in want_s) < 1e-4

"""V1.5 milestone 5: 1F1B is a pure reordering -- same gradients, less memory."""

import pytest
import torch

from meshtrain.experiments.correctness import compare_gradients
from meshtrain.runtime.local import LocalStage, global_param_names, reference_step, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings

MLP = {"type": "mlp"}
TRANSFORMER = {"type": "tiny_transformer", "layers": 4, "hidden_size": 64, "heads": 4, "vocab_size": 64,
               "seq_len": 16}
SPLITS = {
    2: [LocalStage((0, 3)), LocalStage((3, 7))],
    3: [LocalStage((0, 2)), LocalStage((2, 5)), LocalStage((5, 7))],
    4: [LocalStage((0, 2)), LocalStage((2, 4)), LocalStage((4, 6)), LocalStage((6, 7))],
}


def _grads(stages, results):
    out = {}
    for r in results:
        out.update(global_param_names(stages, r.gradients, r.stage_index))
    return out


@pytest.mark.parametrize("S,M,async_transport", [(2, 4, False), (2, 8, True), (3, 8, True), (4, 16, True),
                                                 (4, 2, True)])
def test_1f1b_gradients_match_single_process(S, M, async_transport):
    stages = SPLITS[S]
    s = PipelineSettings("f", steps=1, batch_size=32, num_microbatches=M, schedule="1f1b",
                         async_transport=async_transport, capture_gradients_at_step=0)
    _, ref, _ = reference_step(MLP, s)
    rows = compare_gradients(ref, _grads(stages, run_local_pipeline(MLP, stages, s, transport="tcp")))
    assert max(r.relative for r in rows) < 1e-6, max(rows, key=lambda r: r.relative)


def test_gpipe_and_1f1b_gradients_are_equivalent():
    stages = SPLITS[3]
    common = dict(steps=1, batch_size=32, num_microbatches=8, async_transport=True, capture_gradients_at_step=0)
    g = _grads(stages, run_local_pipeline(MLP, stages, PipelineSettings("g", schedule="gpipe", **common)))
    f = _grads(stages, run_local_pipeline(MLP, stages, PipelineSettings("f", schedule="1f1b", **common)))
    rows = compare_gradients(g, f)
    assert max(r.max_abs for r in rows) < 1e-7


def test_1f1b_training_trajectory_matches_gpipe_and_reference():
    stages = SPLITS[3]
    common = dict(steps=8, batch_size=32, num_microbatches=8, async_transport=True)
    sg, sf = PipelineSettings("g", schedule="gpipe", **common), PipelineSettings("f", schedule="1f1b", **common)
    rg = run_local_pipeline(MLP, stages, sg, optimizer="adam", lr=1e-3, capture_params=True)
    rf = run_local_pipeline(MLP, stages, sf, optimizer="adam", lr=1e-3, capture_params=True)
    ref_losses, _, ref_params = reference_step(MLP, sf, optimizer="adam", lr=1e-3, steps=8)
    assert rf[0].losses == pytest.approx(rg[0].losses, rel=1e-6)
    assert rf[0].losses == pytest.approx(ref_losses, rel=1e-5)
    final = {}
    for r in rf:
        final.update(global_param_names(stages, r.final_params, r.stage_index))
    for k, v in ref_params.items():
        assert torch.allclose(final[k], v, atol=1e-5, rtol=1e-5), k


def test_1f1b_transformer_gradients():
    stages = [LocalStage((0, 2)), LocalStage((2, 4)), LocalStage((4, 6))]
    s = PipelineSettings("t", steps=1, batch_size=8, num_microbatches=4, schedule="1f1b", async_transport=True,
                         capture_gradients_at_step=0)
    _, ref, _ = reference_step(TRANSFORMER, s)
    rows = compare_gradients(ref, _grads(stages, run_local_pipeline(TRANSFORMER, stages, s)))
    assert max(r.relative for r in rows) < 1e-4


def test_1f1b_bounds_saved_activations_per_stage():
    stages = SPLITS[3]
    S, M = 3, 8
    common = dict(steps=2, batch_size=32, num_microbatches=M, async_transport=True)
    rg = run_local_pipeline(MLP, stages, PipelineSettings("g", schedule="gpipe", **common))
    rf = run_local_pipeline(MLP, stages, PipelineSettings("f", schedule="1f1b", **common))
    for r in rf:
        assert r.step_metrics[-1]["memory"]["saved_microbatches_peak"] <= min(S - r.stage_index, M)
    assert rg[0].step_metrics[-1]["memory"]["saved_microbatches_peak"] == M
    g0 = rg[0].step_metrics[-1]["memory"]["saved_activations_peak"]
    f0 = rf[0].step_metrics[-1]["memory"]["saved_activations_peak"]
    assert f0 < g0 * 0.5, (f0, g0)   # stage 0 keeps 3 of 8 microbatches instead of 8


def test_max_inflight_cap_is_respected():
    stages = SPLITS[4]
    s = PipelineSettings("cap", steps=1, batch_size=32, num_microbatches=8, schedule="1f1b",
                         max_inflight_microbatches=2, async_transport=True, capture_gradients_at_step=0)
    res = run_local_pipeline(MLP, stages, s)
    assert all(r.step_metrics[-1]["memory"]["saved_microbatches_peak"] <= 2 for r in res)
    _, ref, _ = reference_step(MLP, s)
    assert max(r.relative for r in compare_gradients(ref, _grads(stages, res))) < 1e-6


def test_stress_four_stages_many_microbatches_many_steps():
    """No deadlocks, no stale contexts, bounded queues, exact training."""
    stages = SPLITS[4]
    s = PipelineSettings("stress", steps=6, batch_size=64, num_microbatches=32, schedule="1f1b",
                         async_transport=True, timeout_s=60)
    res = run_local_pipeline(MLP, stages, s, optimizer="adam", lr=1e-3, link_emulation=(200e6, 0.0002))
    ref_losses, _, _ = reference_step(MLP, s, optimizer="adam", lr=1e-3, steps=6)
    assert res[0].losses == pytest.approx(ref_losses, rel=1e-5)
    for r in res:
        for m in r.step_metrics:
            assert m["outbound_queue_peak"] <= 2 * 32 + 4
            assert m["inbound_queue_peak"] <= 4 * 32 + 16
            assert m["memory"]["saved_microbatches_peak"] <= 4 - r.stage_index
        allocs = [m["buffer_pool"]["allocation_count"] for m in r.step_metrics]
        assert allocs[-1] == allocs[1]  # bounded memory growth: no new buffers after warm-up


def test_pipeline_benchmark_reports_all_modes(tmp_path):
    from meshtrain.experiments.pipeline_bench import run_local_benchmark

    cfg = {"type": "tiny_transformer", "layers": 2, "hidden_size": 64, "heads": 4, "vocab_size": 64, "seq_len": 16}
    r = run_local_benchmark(cfg, num_stages=2, batch_size=8, num_microbatches=4, steps=3, bandwidth_mbps=50,
                            runs_dir=str(tmp_path), write_doc=False)
    modes = [row["mode"] for row in r["rows"]]
    assert modes[0].startswith("V1:") and modes[-1].startswith("V1.5:")
    assert all(row["loss_matches_v1"] in (None, True) for row in r["rows"])
    v1, v15 = r["rows"][0], r["rows"][-1]
    assert v15["peak_saved_mb_stage0"] <= v1["peak_saved_mb_stage0"]
    import os
    assert os.path.exists(os.path.join(r["run_dir"], "timeline-1f1b-async.json"))


def test_tensor_store_reported_and_no_leaked_activations():
    stages = SPLITS[2]
    s = PipelineSettings("ts", steps=3, batch_size=16, num_microbatches=4, schedule="1f1b", async_transport=True)
    res = run_local_pipeline(MLP, stages, s)
    for r in res:
        ts = r.step_metrics[-1]["memory"]["tensor_store"]
        assert ts["parameter"] > 0 and ts["gradient"] == ts["parameter"]
        assert ts["activation"] == 0  # every microbatch's activation entry released after its backward

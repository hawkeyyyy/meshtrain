"""Milestone 1: two (and three) CPU stages in separate processes must produce
the same gradients as a single-process PyTorch model."""

import pytest
import torch

from meshtrain.experiments.correctness import compare_gradients, format_gradient_report, run_gradient_equivalence
from meshtrain.runtime.local import LocalStage, global_param_names, reference_step, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings

MLP = {"type": "mlp"}  # Linear(128,512) GELU Linear(512,512) GELU Linear(512,256) GELU Linear(256,10)
CPU_FP32_TOL = 1e-5  # relative error tolerance for fp32 CPU (only summation-order differences)


def _check(rows):
    print("\n" + format_gradient_report(rows))
    for r in rows:
        assert r.relative < CPU_FP32_TOL, r
        assert r.max_abs < 1e-5, r


def test_two_process_gradients_match_single_process():
    rows, info = run_gradient_equivalence(MLP, [LocalStage((0, 3)), LocalStage((3, 7))], batch_size=32)
    assert {r.stage for r in rows} == {0, 1}
    assert len(rows) == 8  # 4 Linear layers x (weight, bias)
    assert info["distributed_loss"] == pytest.approx(info["reference_loss"], rel=1e-6)
    _check(rows)


def test_three_stage_gradients_match():
    rows, _ = run_gradient_equivalence(MLP, [LocalStage((0, 2)), LocalStage((2, 5)), LocalStage((5, 7))])
    assert {r.stage for r in rows} == {0, 1, 2}
    _check(rows)


def test_microbatched_gradients_match_full_batch_accumulation():
    rows, info = run_gradient_equivalence(MLP, [LocalStage((0, 3)), LocalStage((3, 7))], batch_size=32,
                                          num_microbatches=4)
    _check(rows)


def test_microbatched_gradients_match_unsplit_batch():
    """Accumulating M scaled microbatch losses == full-batch mean loss gradient."""
    settings_full = PipelineSettings("ref", steps=1, batch_size=32, num_microbatches=1)
    _, full_grads, _ = reference_step(MLP, settings_full)
    stages = [LocalStage((0, 4)), LocalStage((4, 7))]
    settings = PipelineSettings("mb", steps=1, batch_size=32, num_microbatches=8, capture_gradients_at_step=0)
    results = run_local_pipeline(MLP, stages, settings)
    dist = {}
    for r in results:
        dist.update(global_param_names(stages, r.gradients, r.stage_index))
    for row in compare_gradients(full_grads, dist):
        assert row.relative < CPU_FP32_TOL, row


def test_transformer_gradients_match():
    cfg = {"type": "tiny_transformer", "layers": 4, "hidden_size": 64, "heads": 4, "vocab_size": 64,
           "seq_len": 16}
    rows, info = run_gradient_equivalence(cfg, [LocalStage((0, 2)), LocalStage((2, 4)), LocalStage((4, 6))],
                                          batch_size=8, num_microbatches=2)
    print("\n" + format_gradient_report(rows))
    for r in rows:
        assert r.relative < 1e-4, r


def test_both_stages_update_and_loss_decreases():
    stages = [LocalStage((0, 3)), LocalStage((3, 7))]
    settings = PipelineSettings("train", steps=60, batch_size=64, num_microbatches=2, log_every=20)
    results = run_local_pipeline(MLP, stages, settings, optimizer="adam", lr=1e-3, capture_params=True)
    losses = results[0].losses
    assert losses[-1] < losses[0] * 0.6, losses
    for r in results:
        changed = [not torch.equal(r.initial_params[k], r.final_params[k]) for k in r.initial_params]
        assert all(changed), f"stage {r.stage_index} has unchanged parameters"
    # Distributed training trajectory matches single-process training.
    ref_losses, _, ref_params = reference_step(MLP, settings, optimizer="adam", lr=1e-3, steps=60)
    assert losses == pytest.approx(ref_losses, rel=1e-4, abs=1e-5)
    final = {}
    for r in results:
        final.update(global_param_names(stages, r.final_params, r.stage_index))
    for k in ref_params:
        assert torch.allclose(final[k], ref_params[k], atol=1e-4, rtol=1e-4), k

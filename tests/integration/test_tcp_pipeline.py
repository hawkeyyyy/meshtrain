"""Milestone 2: the same pipeline over real TCP sockets (loopback here; the
transport is identical when the peer is another machine)."""

import pytest
import torch

from meshtrain.experiments.correctness import run_gradient_equivalence
from meshtrain.runtime.local import LocalStage, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings

MLP = {"type": "mlp"}


def test_tcp_forward_backward_gradients_match():
    rows, info = run_gradient_equivalence(MLP, [LocalStage((0, 3)), LocalStage((3, 7))], transport="tcp")
    assert info["distributed_loss"] == pytest.approx(info["reference_loss"], rel=1e-6)
    assert max(r.relative for r in rows) < 1e-5


def test_tcp_three_stage_microbatched_training():
    stages = [LocalStage((0, 2)), LocalStage((2, 5)), LocalStage((5, 7))]
    settings = PipelineSettings("tcp3", steps=40, batch_size=64, num_microbatches=4, log_every=10)
    results = run_local_pipeline(MLP, stages, settings, optimizer="adam", lr=1e-3, transport="tcp",
                                 capture_params=True)
    losses = results[0].losses
    assert losses[-1] < losses[0] * 0.7
    for r in results:
        assert all(not torch.equal(r.initial_params[k], r.final_params[k]) for k in r.initial_params)
        assert all(m["bytes_sent"] > 0 for m in r.step_metrics)

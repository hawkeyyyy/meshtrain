"""Milestones 3 & 4 on real accelerators. Skipped when the device is absent.

Single-host variants: stages on CUDA/MPS in separate processes talking TCP.
Multi-machine runs use `meshtrain train configs/two_cuda.yaml` (see README).
"""

import pytest

from meshtrain.experiments.correctness import run_gradient_equivalence
from meshtrain.runtime.local import LocalStage

MLP = {"type": "mlp"}
# CUDA/MPS kernels (TF32, different reduction orders) differ from CPU fp32.
ACCEL_REL_TOL = 5e-3


@pytest.mark.cuda
def test_cpu_to_cuda_pipeline_gradients():
    rows, _ = run_gradient_equivalence(MLP, [LocalStage((0, 3), "cpu"), LocalStage((3, 7), "cuda")], transport="tcp")
    assert max(r.relative for r in rows) < ACCEL_REL_TOL


@pytest.mark.cuda
def test_cuda_to_cuda_pipeline_gradients():
    rows, _ = run_gradient_equivalence(MLP, [LocalStage((0, 3), "cuda"), LocalStage((3, 7), "cuda")],
                                       transport="tcp", num_microbatches=2)
    assert max(r.relative for r in rows) < ACCEL_REL_TOL


@pytest.mark.mps
def test_cpu_to_mps_pipeline_gradients():
    rows, _ = run_gradient_equivalence(MLP, [LocalStage((0, 3), "cpu"), LocalStage((3, 7), "mps")], transport="tcp")
    assert max(r.relative for r in rows) < ACCEL_REL_TOL


@pytest.mark.mps
def test_cpu_cpu_mps_three_stage_gradients():
    stages = [LocalStage((0, 2), "cpu"), LocalStage((2, 5), "cpu"), LocalStage((5, 7), "mps")]
    rows, _ = run_gradient_equivalence(MLP, stages, transport="tcp", num_microbatches=2)
    assert max(r.relative for r in rows) < ACCEL_REL_TOL


def _grads(stages, results):
    from meshtrain.runtime.local import global_param_names

    out = {}
    for r in results:
        out.update(global_param_names(stages, r.gradients, r.stage_index))
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_cuda_cuda_async_pipeline_matches_reference(schedule):
    from meshtrain.experiments.correctness import compare_gradients
    from meshtrain.runtime.local import reference_step, run_local_pipeline
    from meshtrain.runtime.pipeline import PipelineSettings

    stages = [LocalStage((0, 3), "cuda"), LocalStage((3, 7), "cuda")]
    s = PipelineSettings("cuda-async", steps=1, batch_size=32, num_microbatches=8, schedule=schedule,
                         async_transport=True, pinned_memory=True, capture_gradients_at_step=0)
    res = run_local_pipeline(MLP, stages, s, transport="tcp")
    _, ref, _ = reference_step(MLP, s)
    assert max(r.relative for r in compare_gradients(ref, _grads(stages, res))) < ACCEL_REL_TOL
    pool = res[1].step_metrics[-1]["buffer_pool"]
    assert pool["pinned"] and pool["pinned_fallbacks"] == 0


@pytest.mark.cuda
def test_cuda_overlap_is_measured_under_emulated_link():
    from meshtrain.runtime.local import run_local_pipeline
    from meshtrain.runtime.pipeline import PipelineSettings

    cfg = {"type": "tiny_transformer", "layers": 6, "hidden_size": 512, "heads": 8, "vocab_size": 1024,
           "seq_len": 128}
    stages = [LocalStage((0, 4), "cuda"), LocalStage((4, 8), "cuda")]
    s = PipelineSettings("ov", steps=4, batch_size=16, num_microbatches=8, schedule="1f1b", async_transport=True)
    res = run_local_pipeline(cfg, stages, s, transport="tcp", link_emulation=(100e6 / 8, 0.001))
    assert sum(m["overlapped_s"] for r in res for m in r.step_metrics[1:]) > 0

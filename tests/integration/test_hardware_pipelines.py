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

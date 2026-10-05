import pytest
import torch

from meshtrain.models import MLPSpec
from meshtrain.runtime.distributed_autograd import BoundaryError, MicrobatchContext
from meshtrain.runtime.stage import Stage


def _stage(index=1, num=2):
    spec = MLPSpec(seed=3)
    return Stage(spec.build_stage(2, 5), stage_index=index, num_stages=num, loss_fn=spec.loss_fn), spec


def test_out_of_order_gradients_reach_the_right_microbatch():
    torch.manual_seed(0)
    inputs = [torch.randn(4, 512) for _ in range(4)]

    def run(order):
        stage, _ = _stage()
        for mb, x in enumerate(inputs):
            stage.forward(x, MicrobatchContext(7, mb))
        returned = {}
        for mb in order:
            g_in, _ = stage.backward(_g(stage, mb), (7, mb))
            returned[mb] = g_in.clone()
        return stage.named_gradients(), returned

    def _g(stage, mb):
        return torch.full((4, 256), float(mb + 1))  # distinct per microbatch

    g_fwd, r_fwd = run([0, 1, 2, 3])
    g_rev, r_rev = run([3, 1, 0, 2])
    for k in g_fwd:
        assert torch.allclose(g_fwd[k], g_rev[k], atol=1e-6)
    for mb in range(4):
        assert torch.allclose(r_fwd[mb], r_rev[mb])
    assert not torch.allclose(r_fwd[0], r_fwd[1])  # routing matters


def test_unknown_microbatch_gradient_rejected():
    stage, _ = _stage()
    stage.forward(torch.randn(2, 512), MicrobatchContext(0, 0))
    with pytest.raises(BoundaryError, match="unknown"):
        stage.backward(torch.randn(2, 512), (0, 1))


def test_duplicate_forward_rejected():
    stage, _ = _stage()
    stage.forward(torch.randn(2, 512), MicrobatchContext(0, 0))
    with pytest.raises(BoundaryError, match="duplicate"):
        stage.forward(torch.randn(2, 512), MicrobatchContext(0, 0))


def test_optimizer_step_refused_with_pending_microbatches():
    stage, _ = _stage()
    stage.forward(torch.randn(2, 512), MicrobatchContext(0, 0))
    with pytest.raises(BoundaryError, match="pending"):
        stage.optimizer_step()


def test_gradient_shape_mismatch_rejected():
    stage, _ = _stage()
    stage.forward(torch.randn(2, 512), MicrobatchContext(0, 0))
    with pytest.raises(BoundaryError, match="shape"):
        stage.backward(torch.randn(3, 512), (0, 0))

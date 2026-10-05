"""V1.5 milestone 3: async transport keeps exact training semantics."""

import random
import threading

import pytest
import torch

from meshtrain.experiments.correctness import compare_gradients
from meshtrain.models import MLPSpec
from meshtrain.networking.tcp import TCPListener, connect
from meshtrain.networking.transport import Transport, TransportTimeout
from meshtrain.runtime.local import LocalStage, global_param_names, reference_step, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineError, PipelineSettings, run_stage
from meshtrain.runtime.stage import Stage
from meshtrain.telemetry import EventLogger

MLP = {"type": "mlp"}


def _dist_grads(stages, results):
    out = {}
    for r in results:
        out.update(global_param_names(stages, r.gradients, r.stage_index))
    return out


@pytest.mark.parametrize("transport", ["pipe", "tcp"])
def test_async_gpipe_gradients_match_reference(transport):
    stages = [LocalStage((0, 2)), LocalStage((2, 5)), LocalStage((5, 7))]
    s = PipelineSettings("a", steps=1, batch_size=32, num_microbatches=4, async_transport=True,
                         capture_gradients_at_step=0)
    _, ref, _ = reference_step(MLP, s)
    rows = compare_gradients(ref, _dist_grads(stages, run_local_pipeline(MLP, stages, s, transport=transport)))
    assert max(r.relative for r in rows) < 1e-6


def test_async_with_emulated_link_trains_identically():
    stages = [LocalStage((0, 3)), LocalStage((3, 7))]
    s_sync = PipelineSettings("s", steps=5, batch_size=32, num_microbatches=4)
    s_async = PipelineSettings("a", steps=5, batch_size=32, num_microbatches=4, async_transport=True)
    a = run_local_pipeline(MLP, stages, s_sync, optimizer="adam", lr=1e-3, link_emulation=(20e6, 0.001))
    b = run_local_pipeline(MLP, stages, s_async, optimizer="adam", lr=1e-3, link_emulation=(20e6, 0.001))
    assert a[0].losses == pytest.approx(b[0].losses, rel=1e-6)
    # blocking sends can never overlap compute; async ones are measured
    assert all(m["overlapped_s"] < 1e-3 for r in a for m in r.step_metrics)
    assert all(m["communication_s"] > 0 for r in b for m in r.step_metrics)


class ShufflingTransport(Transport):
    """Delivers received packets in a shuffled order (within a small window)."""

    def __init__(self, inner, window=4, seed=0):
        super().__init__(inner.max_payload_bytes)
        self.inner, self.window, self.rng = inner, window, random.Random(seed)
        self.buf = []

    def _send_frame(self, frame):
        self.inner._send_frame(frame)

    def _send_parts(self, head, payload):
        self.inner._send_parts(head, payload)

    def _recv_packet(self, timeout):
        while len(self.buf) < self.window:
            try:
                self.buf.append(self.inner._recv_packet(0.05 if self.buf else timeout))
            except TransportTimeout:
                if self.buf:
                    break
                raise
        i = self.rng.randrange(len(self.buf))
        return self.buf.pop(i)

    def close(self):
        self.inner.close()


def _pair():
    listener = TCPListener("127.0.0.1", 0)
    box = {}
    th = threading.Thread(target=lambda: box.update(s=listener.accept(timeout=5)))
    th.start()
    c = connect("127.0.0.1", listener.port, timeout=5, hello={"job_id": "x"})
    th.join()
    listener.close()
    return c, box["s"][0]


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_out_of_order_arrival_is_routed_by_microbatch(schedule):
    spec = MLPSpec()
    down, up = _pair()
    up = ShufflingTransport(up, window=5, seed=1)      # activations/targets arrive shuffled
    down = ShufflingTransport(down, window=3, seed=2)  # gradients arrive shuffled
    s = PipelineSettings("x", steps=2, batch_size=32, num_microbatches=8, schedule=schedule,
                         async_transport=True, capture_gradients_at_step=0)
    s0 = Stage(spec.build_stage(0, 3), stage_index=0, num_stages=2)
    s1 = Stage(spec.build_stage(3, 7), stage_index=1, num_stages=2, loss_fn=spec.loss_fn)
    out = {}
    th = threading.Thread(target=lambda: out.update(r1=run_stage(
        s1, spec, s, upstream=up, downstream=None, logger=EventLogger("w1", "x", verbose=False))))
    th.start()
    r0 = run_stage(s0, spec, s, upstream=None, downstream=down, logger=EventLogger("w0", "x", verbose=False))
    th.join(30)
    stages = [LocalStage((0, 3)), LocalStage((3, 7))]
    _, ref, _ = reference_step(MLP, s)
    rows = compare_gradients(ref, _dist_grads(stages, [r0, out["r1"]]))
    assert max(r.relative for r in rows) < 1e-5


@pytest.mark.parametrize("async_transport", [False, True])
def test_peer_disconnect_fails_fast_async_and_sync(async_transport):
    spec = MLPSpec()
    down, peer = _pair()
    s = PipelineSettings("f", steps=1, batch_size=8, timeout_s=30.0, async_transport=async_transport)
    threading.Timer(0.5, peer.close).start()
    with pytest.raises(PipelineError, match="disconnected|send to"):
        run_stage(Stage(spec.build_stage(0, 3), stage_index=0, num_stages=2), spec, s, upstream=None,
                  downstream=down, logger=EventLogger("w0", "f", verbose=False))


def test_stall_produces_state_dump():
    spec = MLPSpec()
    down, peer = _pair()  # peer never answers
    s = PipelineSettings("f", steps=1, batch_size=8, num_microbatches=2, timeout_s=1.0, schedule="1f1b",
                         async_transport=True)
    with pytest.raises(PipelineError, match="stalled") as ei:
        run_stage(Stage(spec.build_stage(0, 3), stage_index=0, num_stages=2), spec, s, upstream=None,
                  downstream=down, logger=EventLogger("w0", "f", verbose=False))
    msg = str(ei.value)
    for key in ("next_action", "saved_contexts", "pending_sends", "microbatches"):
        assert key in msg
    peer.close()

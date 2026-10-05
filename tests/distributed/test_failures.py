"""Failure must be clean: no hangs, meaningful errors."""

import threading
import time

import pytest

from meshtrain.models import MLPSpec
from meshtrain.networking.tcp import TCPListener, connect
from meshtrain.runtime.pipeline import PipelineError, PipelineSettings, run_stage
from meshtrain.runtime.stage import Stage
from meshtrain.telemetry import EventLogger


def _link_pair():
    listener = TCPListener("127.0.0.1", 0)
    box = {}
    th = threading.Thread(target=lambda: box.update(s=listener.accept(timeout=5)))
    th.start()
    c = connect("127.0.0.1", listener.port, timeout=5, hello={"job_id": "f"})
    th.join()
    listener.close()
    return c, box["s"][0]


def _stage0(spec):
    return Stage(spec.build_stage(0, 3), stage_index=0, num_stages=2)


def test_silent_peer_times_out_with_message():
    spec = MLPSpec()
    down, peer = _link_pair()
    settings = PipelineSettings("f", steps=1, batch_size=8, timeout_s=1.0)
    t0 = time.monotonic()
    with pytest.raises(PipelineError, match="timeout"):
        run_stage(_stage0(spec), spec, settings, upstream=None, downstream=down,
                  logger=EventLogger("w0", "f", verbose=False))
    assert time.monotonic() - t0 < 5
    peer.close()


def test_peer_disconnect_fails_fast():
    spec = MLPSpec()
    down, peer = _link_pair()
    settings = PipelineSettings("f", steps=1, batch_size=8, timeout_s=30.0)
    threading.Timer(0.5, peer.close).start()
    t0 = time.monotonic()
    with pytest.raises(PipelineError, match="disconnected"):
        run_stage(_stage0(spec), spec, settings, upstream=None, downstream=down,
                  logger=EventLogger("w0", "f", verbose=False))
    assert time.monotonic() - t0 < 5, "disconnect must not wait for the full timeout"


def test_downstream_crash_is_reported_upstream():
    """A stage that fails sends an ERROR packet so its neighbour stops immediately."""
    spec = MLPSpec()
    down, up = _link_pair()
    settings = PipelineSettings("f", steps=1, batch_size=8, timeout_s=30.0)

    def bad_last_stage():
        # Wrong layer range: layer 6 expects 256 features, receives 512 -> shape error.
        stage = Stage(spec.build_stage(6, 7), stage_index=1, num_stages=2, loss_fn=spec.loss_fn)
        with pytest.raises(Exception):
            run_stage(stage, spec, settings, upstream=up, downstream=None,
                      logger=EventLogger("w1", "f", verbose=False))

    th = threading.Thread(target=bad_last_stage)
    th.start()
    with pytest.raises(PipelineError, match="peer reported error"):
        run_stage(_stage0(spec), spec, settings, upstream=None, downstream=down,
                  logger=EventLogger("w0", "f", verbose=False))
    th.join(5)
    assert not th.is_alive()

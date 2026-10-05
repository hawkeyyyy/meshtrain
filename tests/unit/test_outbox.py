import queue
import threading
import time

import pytest
import torch

from meshtrain.networking.transport import Transport, TransportClosed
from meshtrain.runtime.outbox import Outbox, OutboxError
from meshtrain.runtime.tensor_packet import MessageType
from meshtrain.runtime.timeline import Timeline
from meshtrain.worker.device import CPUDeviceAdapter


class SlowTransport(Transport):
    def __init__(self, delay=0.05, fail_after=None):
        super().__init__()
        self.delay, self.fail_after = delay, fail_after
        self.frames = []

    def _send_frame(self, frame):
        if self.fail_after is not None and len(self.frames) >= self.fail_after:
            raise TransportClosed("peer went away")
        time.sleep(self.delay)
        self.frames.append(frame)

    def _recv_packet(self, timeout):
        raise NotImplementedError

    def close(self):
        pass


def _submit(ob, i):
    ob.submit_tensor("down", torch.full((4,), float(i)), MessageType.FORWARD_ACTIVATION, device=CPUDeviceAdapter(),
                     step=0, microbatch=i, notify=True, job_id="j")


def test_async_submit_returns_before_network_send():
    link = SlowTransport(delay=0.2)
    ob = Outbox({"down": link}, Timeline(), async_mode=True, max_queue=8)
    t0 = time.perf_counter()
    _submit(ob, 0)
    assert time.perf_counter() - t0 < 0.1  # compute thread did not wait for the socket
    ob.flush(timeout=5)
    assert len(link.frames) == 1
    ob.close()


def test_sync_mode_sends_inline():
    link = SlowTransport(delay=0.1)
    ob = Outbox({"down": link}, Timeline(), async_mode=False)
    t0 = time.perf_counter()
    _submit(ob, 0)
    assert time.perf_counter() - t0 >= 0.1 and len(link.frames) == 1


def test_bounded_queue_back_pressure_and_fifo_order():
    link = SlowTransport(delay=0.03)
    events = queue.Queue()
    ob = Outbox({"down": link}, Timeline(), async_mode=True, max_queue=2, events=events)
    for i in range(8):
        _submit(ob, i)
    ob.flush(timeout=5)
    assert ob.peak_depth["down"] <= 2
    done = [events.get_nowait()[3] for _ in range(8)]
    assert done == list(range(8))  # per-link FIFO preserved
    from meshtrain.networking.protocol import decode_packet
    assert [decode_packet(f).microbatch_id for f in link.frames] == list(range(8))
    ob.close()


def test_full_queue_times_out_instead_of_hanging():
    link = SlowTransport(delay=5.0)
    ob = Outbox({"down": link}, Timeline(), async_mode=True, max_queue=1, timeout_s=0.3)
    _submit(ob, 0)  # taken by the sender thread (stuck in a 5 s send)
    _submit(ob, 1)  # fills the queue
    with pytest.raises(OutboxError, match="stayed full"):
        _submit(ob, 2)


def test_send_failure_is_reported_and_raised():
    link = SlowTransport(delay=0.0, fail_after=1)
    events = queue.Queue()
    ob = Outbox({"down": link}, Timeline(), async_mode=True, events=events)
    _submit(ob, 0)
    _submit(ob, 1)
    kinds = []
    deadline = time.time() + 5
    while time.time() < deadline and "send_error" not in kinds:
        try:
            kinds.append(events.get(timeout=0.5)[0])
        except queue.Empty:
            pass
    assert "send_error" in kinds
    with pytest.raises(OutboxError, match="peer went away"):
        _submit(ob, 2)


def test_close_stops_sender_threads():
    before = {t.name for t in threading.enumerate()}
    ob = Outbox({"down": SlowTransport(0.0), "up": SlowTransport(0.0)}, Timeline(), async_mode=True)
    ob.close()
    time.sleep(0.1)
    leftover = {t.name for t in threading.enumerate() if t.name.startswith("send-")} - before
    assert not leftover


def test_zero_copy_payload_matches_tensor():
    link = SlowTransport(delay=0.0)
    ob = Outbox({"down": link}, Timeline(), async_mode=False)
    t = torch.randn(3, 5)
    ob.submit_tensor("down", t, MessageType.FORWARD_ACTIVATION, device=CPUDeviceAdapter(), step=1, microbatch=2)
    from meshtrain.networking.protocol import decode_packet
    from meshtrain.runtime.serialization import packet_to_tensor
    p = decode_packet(link.frames[0])
    assert (p.step_id, p.microbatch_id) == (1, 2) and torch.equal(packet_to_tensor(p), t)

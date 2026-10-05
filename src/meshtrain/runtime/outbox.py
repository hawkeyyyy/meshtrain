"""Outbound tensor path: staging, serialization and (optionally async) sends.

    compute thread                      sender thread (one per link, async mode)
    ──────────────                      ─────────────────────────────────────────
    submit(tensor)  ──► begin_d2h ──►   [bounded FIFO]  ──► finish D2H (wait event)
       (records ordering point)                            serialize (zero-copy)
                                                           TCP send
                                                           SEND_COMPLETED / SEND_ERROR ─► executor

* ``async_mode=False`` reproduces V1: everything runs on the compute thread.
* Ordering is FIFO per link, so per-link message order is preserved; across
  links (upstream vs downstream) messages are independent.
* The queue is bounded (``max_queue``): when the network is slower than
  compute, ``submit`` blocks (back-pressure) instead of growing RAM without
  limit. Submits never block indefinitely: ``timeout_s`` turns a stuck
  link into a ``PipelineError``.
* Network errors on the sender thread are reported to the executor as
  ``("send_error", ...)`` events and re-raised there.

Device specifics (pinned buffers, CUDA streams/events, MPS sync) live behind
``DeviceAdapter.begin_d2h`` -- this module only calls it.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from meshtrain.networking.transport import Transport
from meshtrain.runtime.serialization import tensor_to_bytes
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket
from meshtrain.runtime.timeline import Timeline

_STOP = object()


class OutboxError(RuntimeError):
    pass


@dataclass
class _Item:
    link: str
    packet: TensorPacket | None
    finish: Callable | None  # returns (cpu_tensor, release_fn) when called
    submitted: float
    notify: bool
    fields: dict = field(default_factory=dict)


class Outbox:
    def __init__(self, links: dict[str, Transport], timeline: Timeline, *, async_mode: bool,
                 max_queue: int = 16, timeout_s: float = 60.0,
                 events: "queue.Queue | None" = None):
        self.links = links
        self.timeline = timeline
        self.async_mode = async_mode
        self.timeout_s = timeout_s
        self.events = events  # executor's inbound event queue (send completions/errors)
        self.bytes_sent = {name: 0 for name in links}
        self.peak_depth = {name: 0 for name in links}
        self.sent_messages = 0
        self._queues: dict[str, queue.Queue] = {}
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        if async_mode:
            for name in links:
                q: queue.Queue = queue.Queue(maxsize=max(1, max_queue))
                self._queues[name] = q
                t = threading.Thread(target=self._sender, args=(name, q), daemon=True, name=f"send-{name}")
                t.start()
                self._threads.append(t)

    # -- submission (compute thread) -----------------------------------------
    def submit_tensor(self, link: str, tensor, message_type: MessageType, *, device, step: int, microbatch: int,
                      notify: bool = False, **fields) -> None:
        """Queue a tensor for sending. ``device.begin_d2h`` captures the
        ordering point on the compute thread; the copy may finish later."""
        self._raise_if_failed()
        with self.timeline.span("D2H_COPY", step, microbatch, phase="begin"):
            finish = device.begin_d2h(tensor)
        fields.update(message_type=message_type, step_id=step, microbatch_id=microbatch)
        self._enqueue(_Item(link, None, finish, self.timeline.now(), notify, fields))

    def submit_packet(self, link: str, packet: TensorPacket) -> None:
        """Forward an already-serialized packet (target relay)."""
        self._raise_if_failed()
        self._enqueue(_Item(link, packet, None, self.timeline.now(), False))

    def _enqueue(self, item: _Item) -> None:
        if not self.async_mode:
            self._send(item)
            return
        q = self._queues[item.link]
        try:
            q.put(item, timeout=self.timeout_s)
        except queue.Full:
            raise OutboxError(f"outbound queue to {item.link} stayed full for {self.timeout_s}s "
                              f"(peer not draining)") from None
        self.peak_depth[item.link] = max(self.peak_depth[item.link], q.qsize())

    # -- sending ----------------------------------------------------------------
    def _send(self, item: _Item) -> None:
        tl = self.timeline
        f = item.fields
        step, mb = f.get("step_id", item.packet.step_id if item.packet else None), f.get("microbatch_id")
        start = tl.now()
        if self.async_mode:
            tl.add("QUEUE_WAIT", item.submitted, start, step, mb, link=item.link)
        release = None
        if item.packet is not None:
            packet, views = item.packet, None
        else:
            with tl.span("D2H_COPY", step, mb, phase="finish"):
                cpu, release = item.finish()
            with tl.span("SERIALIZE", step, mb):
                payload, dtype, shape = tensor_to_bytes(cpu, copy=False)
            mtype = f.pop("message_type")
            packet = TensorPacket(mtype, shape=shape, dtype=dtype, payload=payload, **f)
        link = self.links[item.link]
        before = link.bytes_sent
        with tl.span("NETWORK_SEND", step, mb, link=item.link, bytes=len(packet.payload)):
            link.send_packet(packet)
        if release is not None:
            release()
        self.bytes_sent[item.link] += link.bytes_sent - before
        self.sent_messages += 1
        if item.notify and self.events is not None:
            self.events.put(("send_done", item.link, packet.step_id, packet.microbatch_id))

    def _sender(self, name: str, q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is _STOP:
                return
            try:
                self._send(item)
            except BaseException as exc:  # surface to the executor, stop sending on this link
                self._error = exc
                if self.events is not None:
                    self.events.put(("send_error", name, exc))
                return
            finally:
                q.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise OutboxError(f"send failed: {type(self._error).__name__}: {self._error}") from self._error

    # -- lifecycle --------------------------------------------------------------
    def pending(self) -> dict[str, int]:
        return {name: q.qsize() for name, q in self._queues.items()}

    def flush(self, timeout: float | None = None) -> None:
        """Wait until everything queued has been sent (or raise)."""
        if not self.async_mode:
            return
        timeout = self.timeout_s if timeout is None else timeout
        done = threading.Event()

        def waiter():
            for q in self._queues.values():
                q.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        if not done.wait(timeout):
            self._raise_if_failed()
            raise OutboxError(f"outbound queues not drained after {timeout}s: {self.pending()}")
        self._raise_if_failed()

    def close(self) -> None:
        for q in self._queues.values():
            try:
                q.put_nowait(_STOP)
            except queue.Full:
                # Sender is stuck on a dead socket; it is a daemon thread and
                # exits when the link is closed by the caller.
                pass
        for t in self._threads:
            t.join(timeout=2.0)

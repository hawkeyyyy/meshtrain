"""Emulated network links for single-host experiments (NOT used in real runs).

Loopback TCP moves gigabytes per second, which hides every communication
effect V1.5 is meant to study. ``EmulatedLink`` wraps a real transport and
makes each *direction* behave like a slower link:

    send(frame) occupies the sender's side of the wire for
        latency_s + frame_bytes / bandwidth_Bps
    (the real bytes still go over the wrapped transport)

Sends on one direction serialize (one wire), the two directions are
independent (full duplex). The delay is paid by whichever thread sends: the
compute thread with blocking transport (V1 behaviour, fully exposed), or the
sender thread with async transport (can overlap with compute). Latency is
modelled as part of the per-message occupancy, which over-penalizes many
tiny messages slightly; tensors here are large so this is negligible.

Every result produced with emulation must say so (bandwidth/latency used).
"""

from __future__ import annotations

import threading
import time

from meshtrain.networking.transport import Transport


class EmulatedLink(Transport):
    def __init__(self, inner: Transport, bandwidth_Bps: float, latency_s: float = 0.0):
        super().__init__(inner.max_payload_bytes)
        if bandwidth_Bps <= 0:
            raise ValueError("bandwidth must be positive")
        self.inner = inner
        self.bandwidth_Bps = bandwidth_Bps
        self.latency_s = latency_s
        self._wire = threading.Lock()

    @property
    def payload_allocator(self):
        return self.inner.payload_allocator

    @payload_allocator.setter
    def payload_allocator(self, fn):
        if hasattr(self, "inner"):
            self.inner.payload_allocator = fn

    @property
    def last_frame_s(self) -> float:
        return getattr(self.inner, "last_frame_s", 0.0)

    def _occupy(self, nbytes: int, started: float) -> None:
        target = self.latency_s + nbytes / self.bandwidth_Bps
        remaining = target - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _send_frame(self, frame: bytes) -> None:
        self._send_parts(frame, b"")

    def _send_parts(self, head: bytes, payload) -> None:
        with self._wire:
            t0 = time.perf_counter()
            self.inner._send_parts(head, payload)
            self._occupy(len(head) + len(payload), t0)

    def _recv_packet(self, timeout):
        return self.inner._recv_packet(timeout)

    def close(self) -> None:
        self.inner.close()

    @property
    def peer(self) -> str:
        return getattr(self.inner, "peer", "emulated")

    @property
    def frame_timeout_s(self):
        return getattr(self.inner, "frame_timeout_s", None)

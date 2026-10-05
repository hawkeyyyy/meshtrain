"""Point-to-point tensor transports.

``Transport`` is a bidirectional, ordered, reliable channel between two
pipeline peers. It moves *bytes*; it has no knowledge of PyTorch devices. The
runtime converts tensors to/from ``TensorPacket`` (see runtime/serialization)
and hands packets to a transport.

Implementations:

* ``PipeTransport``   -- multiprocessing pipe; separate processes, same machine
                         (Milestone 1, no networking involved).
* ``TCPTransport``    -- raw TCP socket (Milestone 2+), see networking/tcp.py.

Later transports (QUIC, RDMA, compressed links) only need ``send_packet``
and ``recv_packet``.
"""

from __future__ import annotations

import abc
import threading
import time

import torch

from meshtrain.networking.protocol import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    PREFIX_SIZE,
    ProtocolError,
    decode_body,
    decode_packet,
    encode_head,
    parse_prefix,
)
from meshtrain.runtime.serialization import TensorLifecycle, packet_to_tensor, tensor_to_packet
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket


class TransportError(RuntimeError):
    pass


class TransportTimeout(TransportError, TimeoutError):
    pass


class TransportClosed(TransportError, ConnectionError):
    pass


class Transport(abc.ABC):
    """Abstract bidirectional packet channel with byte accounting."""

    def __init__(self, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES):
        self.max_payload_bytes = max_payload_bytes
        self.bytes_sent = 0
        self.bytes_received = 0
        self._send_lock = threading.Lock()

    # -- raw frames -------------------------------------------------------
    @abc.abstractmethod
    def _send_frame(self, frame: bytes) -> None: ...

    def _send_parts(self, head: bytes, payload) -> None:
        """Send a frame given as header + payload (override for zero-copy)."""
        self._send_frame(head + bytes(payload))

    @abc.abstractmethod
    def _recv_packet(self, timeout: float | None) -> tuple[TensorPacket, int]: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    # -- packets ----------------------------------------------------------
    def send_packet(self, packet: TensorPacket) -> float:
        """Send a packet; returns seconds spent in the network send."""
        if len(packet.payload) > self.max_payload_bytes:
            raise ProtocolError(f"payload {len(packet.payload)} exceeds limit {self.max_payload_bytes}")
        head = encode_head(packet)
        t0 = time.perf_counter()
        with self._send_lock:
            self._send_parts(head, packet.payload)
        self.bytes_sent += len(head) + len(packet.payload)
        return time.perf_counter() - t0

    def recv_packet(self, timeout: float | None = None) -> TensorPacket:
        packet, nbytes = self._recv_packet(timeout)
        self.bytes_received += nbytes
        return packet

    # -- tensors (convenience) -------------------------------------------
    def send_tensor(
        self,
        tensor: torch.Tensor,
        message_type: MessageType,
        *,
        device=None,
        lifecycle: TensorLifecycle | None = None,
        **fields,
    ) -> TensorLifecycle:
        lc = lifecycle if lifecycle is not None else TensorLifecycle()
        packet = tensor_to_packet(tensor, message_type, device=device, lifecycle=lc, **fields)
        lc.add("network_send", self.send_packet(packet))
        return lc

    def recv_tensor(self, timeout: float | None = None, *, device=None) -> tuple[torch.Tensor, TensorPacket]:
        lc = TensorLifecycle()
        t0 = time.perf_counter()
        packet = self.recv_packet(timeout)
        lc.add("network_recv", time.perf_counter() - t0)
        return packet_to_tensor(packet, device=device, lifecycle=lc), packet


class PipeTransport(Transport):
    """Transport over a ``multiprocessing.connection.Connection`` (same host)."""

    def __init__(self, conn, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES):
        super().__init__(max_payload_bytes)
        self.conn = conn

    def _send_frame(self, frame: bytes) -> None:
        try:
            self.conn.send_bytes(frame)
        except (BrokenPipeError, OSError, EOFError) as exc:
            raise TransportClosed(f"peer closed pipe: {exc}") from exc

    def _recv_packet(self, timeout):
        try:
            ready = self.conn.poll(timeout)
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise TransportClosed(f"peer closed pipe: {exc}") from exc
        if not ready:
            raise TransportTimeout(f"no packet within {timeout}s")
        try:
            frame = self.conn.recv_bytes(PREFIX_SIZE + 64 * 1024 + self.max_payload_bytes)
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise TransportClosed(f"peer closed pipe: {exc}") from exc
        except OSError as exc:
            raise ProtocolError(f"pipe receive failed: {exc}") from exc
        return decode_packet(frame, self.max_payload_bytes), len(frame)

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass


__all__ = [
    "Transport",
    "PipeTransport",
    "TransportError",
    "TransportTimeout",
    "TransportClosed",
    "ProtocolError",
    "parse_prefix",
    "decode_body",
]

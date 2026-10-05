"""Raw TCP data plane.

Each pipeline link is one TCP connection carrying length-prefixed frames
(see protocol.py) in both directions. The connecting side sends a CONTROL
"hello" packet first (job id, stage index, cluster token) so a worker's single
data-plane listener can route incoming connections.
"""

from __future__ import annotations

import select
import socket
import time

from meshtrain.networking.protocol import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    PREFIX_SIZE,
    ProtocolError,
    decode_body,
    parse_prefix,
)
from meshtrain.networking.transport import Transport, TransportClosed, TransportTimeout
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        try:
            k = sock.recv_into(view[got:], min(n - got, 4 * 1024 * 1024))
        except socket.timeout as exc:
            raise TransportTimeout(f"peer stalled mid-frame ({got}/{n} bytes)") from exc
        except OSError as exc:
            raise TransportClosed(f"connection error: {exc}") from exc
        if k == 0:
            raise TransportClosed("peer closed the connection")
        got += k
    return bytes(buf)


class TCPTransport(Transport):
    def __init__(self, sock: socket.socket, *, frame_timeout_s: float = 60.0,
                 max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES, peer: str = ""):
        super().__init__(max_payload_bytes)
        self.sock = sock
        self.frame_timeout_s = frame_timeout_s
        self.peer = peer or "%s:%s" % sock.getpeername()[:2]
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(frame_timeout_s)

    def _send_frame(self, frame: bytes) -> None:
        try:
            self.sock.sendall(frame)
        except socket.timeout as exc:
            raise TransportTimeout(f"send to {self.peer} timed out") from exc
        except OSError as exc:
            raise TransportClosed(f"send to {self.peer} failed: {exc}") from exc

    def _recv_packet(self, timeout):
        # Wait for the *start* of a frame with ``timeout``; once a frame has
        # started, read all of it (bounded by frame_timeout_s) so we never
        # abandon a half-read frame.
        try:
            ready, _, _ = select.select([self.sock], [], [], timeout)
        except (OSError, ValueError) as exc:
            raise TransportClosed(f"connection to {self.peer} closed: {exc}") from exc
        if not ready:
            raise TransportTimeout(f"no packet from {self.peer} within {timeout}s")
        prefix = _recv_exact(self.sock, PREFIX_SIZE)
        header_len, payload_len = parse_prefix(prefix, self.max_payload_bytes)
        header = _recv_exact(self.sock, header_len)
        payload = _recv_exact(self.sock, payload_len) if payload_len else b""
        return decode_body(header, payload), PREFIX_SIZE + header_len + payload_len

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def _hello_packet(hello: dict) -> TensorPacket:
    return TensorPacket(MessageType.CONTROL, job_id=str(hello.get("job_id", "")),
                        meta={"command": "HELLO", **hello})


def connect(host: str, port: int, *, timeout: float = 30.0, hello: dict | None = None,
            frame_timeout_s: float = 60.0, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> TCPTransport:
    """Connect (retrying until ``timeout``: the peer may still be starting) and send hello."""
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=min(5.0, timeout))
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(0.1)
    else:
        raise TransportTimeout(f"could not connect to {host}:{port} within {timeout}s: {last_exc}")
    t = TCPTransport(sock, frame_timeout_s=frame_timeout_s, max_payload_bytes=max_payload_bytes,
                     peer=f"{host}:{port}")
    t.send_packet(_hello_packet(hello or {}))
    return t


class TCPListener:
    def __init__(self, host: str = "0.0.0.0", port: int = 0, *, token: str | None = None,
                 frame_timeout_s: float = 60.0, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.token = token
        self.frame_timeout_s = frame_timeout_s
        self.max_payload_bytes = max_payload_bytes

    def accept(self, timeout: float | None = None) -> tuple[TCPTransport, dict]:
        """Accept one connection and read its hello. Rejects bad hellos/tokens."""
        self.sock.settimeout(timeout)
        try:
            conn, addr = self.sock.accept()
        except socket.timeout as exc:
            raise TransportTimeout(f"no peer connected within {timeout}s") from exc
        t = TCPTransport(conn, frame_timeout_s=self.frame_timeout_s, max_payload_bytes=self.max_payload_bytes,
                         peer=f"{addr[0]}:{addr[1]}")
        try:
            hello = t.recv_packet(timeout=min(timeout or 10.0, 10.0))
            if hello.message_type != MessageType.CONTROL or hello.meta.get("command") != "HELLO":
                raise ProtocolError("first packet must be a HELLO control packet")
            if self.token is not None and hello.meta.get("token") != self.token:
                raise ProtocolError("invalid cluster token")
        except Exception:
            t.close()
            raise
        return t, dict(hello.meta)

    def close(self) -> None:
        self.sock.close()

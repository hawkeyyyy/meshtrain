"""Worker-to-worker latency / bandwidth probes over the data plane.

Measurements are directional: ``i -> j`` is measured by worker i sending to
worker j's data-plane listener, so asymmetric links show up as
``bandwidth[i][j] != bandwidth[j][i]``.

Probe protocol (after the HELLO with ``command=PROBE``):
    client: CONTROL{op=PING}               server: ACK
    client: FORWARD_ACTIVATION(blob) x N   (no reply)
    client: CONTROL{op=SYNC}               server: ACK{bytes=<received>}
    client: CONTROL{op=BYE}
"""

from __future__ import annotations

import statistics
import time

import torch

from meshtrain.networking.tcp import connect
from meshtrain.networking.transport import Transport
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket


def serve_probe(link: Transport, timeout: float = 30.0) -> None:
    received = 0
    while True:
        p = link.recv_packet(timeout=timeout)
        if p.message_type == MessageType.CONTROL:
            op = p.meta.get("op")
            if op == "PING":
                link.send_packet(TensorPacket(MessageType.ACK, meta={"op": "PONG"}))
            elif op == "SYNC":
                link.send_packet(TensorPacket(MessageType.ACK, meta={"bytes": received}))
                received = 0
            elif op == "BYE":
                return
        elif p.has_tensor:
            received += len(p.payload)


def measure_link(host: str, port: int, *, token: str | None = None, pings: int = 10,
                 payload_mb: float = 16.0, chunks: int = 4, timeout: float = 30.0) -> dict:
    hello = {"probe": True, "command_kind": "PROBE"}
    if token is not None:
        hello["token"] = token
    link = connect(host, port, timeout=timeout, hello=hello, frame_timeout_s=timeout)
    try:
        rtts = []
        for _ in range(pings):
            t0 = time.perf_counter()
            link.send_packet(TensorPacket(MessageType.CONTROL, meta={"op": "PING"}))
            link.recv_packet(timeout=timeout)
            rtts.append(time.perf_counter() - t0)
        chunk_elems = max(1, int(payload_mb * 1e6 / 4 / chunks))
        blob = torch.zeros(chunk_elems)
        t0 = time.perf_counter()
        for _ in range(chunks):
            link.send_tensor(blob, MessageType.FORWARD_ACTIVATION)
        link.send_packet(TensorPacket(MessageType.CONTROL, meta={"op": "SYNC"}))
        ack = link.recv_packet(timeout=timeout)
        elapsed = time.perf_counter() - t0
        nbytes = int(ack.meta.get("bytes", 0))
        link.send_packet(TensorPacket(MessageType.CONTROL, meta={"op": "BYE"}))
    finally:
        link.close()
    rtt = statistics.median(rtts)
    # Subtract one RTT (the SYNC round trip) from the transfer time.
    transfer = max(elapsed - rtt, 1e-9)
    return {
        "latency_s": rtt / 2,
        "rtt_s": rtt,
        "bandwidth_Bps": nbytes / transfer,
        "bandwidth_Mbps": nbytes * 8 / transfer / 1e6,
        "bytes": nbytes,
    }


def format_matrix(names: list[str], values: dict[tuple[str, str], float], fmt) -> str:
    width = max(10, *(len(n) + 2 for n in names))
    lines = [" " * width + "".join(f"{n:>{width}}" for n in names)]
    for i in names:
        row = f"{i:<{width}}"
        for j in names:
            row += f"{'-':>{width}}" if i == j else f"{fmt(values.get((i, j))):>{width}}"
        lines.append(row)
    return "\n".join(lines)

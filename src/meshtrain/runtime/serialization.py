"""Tensor <-> bytes conversion and the timed tensor lifecycle.

Lifecycle (each phase is timed, see docs/architecture.md):

    CREATE -> LOCAL_DEVICE -> DETACH -> CPU_STAGING -> SERIALIZE -> NETWORK
           -> DESERIALIZE -> TARGET_DEVICE -> COMPUTE -> GRADIENT -> reverse

Serialization is intentionally simple: detach, copy to CPU, make contiguous,
take the raw bytes. Deserialization rebuilds a CPU tensor from the bytes plus
the dtype/shape header and lets the device adapter move it to its device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from meshtrain.runtime.tensor_packet import DTYPE_TO_NAME, NAME_TO_DTYPE, MessageType, TensorPacket

if TYPE_CHECKING:  # pragma: no cover
    from meshtrain.worker.device import DeviceAdapter


@dataclass
class TensorLifecycle:
    """Accumulates per-phase timings (seconds) for one tensor transfer."""

    timings: dict[str, float] = field(default_factory=dict)
    nbytes: int = 0

    def add(self, phase: str, seconds: float) -> None:
        self.timings[phase] = self.timings.get(phase, 0.0) + seconds

    def timed(self, phase: str):
        lifecycle = self

        class _Timer:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                lifecycle.add(phase, time.perf_counter() - self_inner.t0)

        return _Timer()


def tensor_to_bytes(tensor: torch.Tensor, copy: bool = True) -> tuple[bytes | memoryview, str, tuple[int, ...]]:
    """Return (raw bytes, dtype name, shape) for a CPU tensor.

    ``copy=False`` returns a zero-copy ``memoryview`` of the tensor's memory:
    the caller must keep the tensor alive and unmodified until the bytes have
    been sent (the async outbox does).
    """
    if tensor.device.type != "cpu":
        raise ValueError("tensor_to_bytes expects a CPU tensor; stage it first")
    if tensor.dtype not in DTYPE_TO_NAME:
        raise ValueError(f"dtype {tensor.dtype} is not supported on the wire")
    t = tensor.detach().contiguous()
    shape = tuple(t.shape)
    if t.numel() == 0:
        return b"", DTYPE_TO_NAME[t.dtype], shape
    # Reinterpret as bytes; works for bfloat16/bool too (numpy has no bf16).
    arr = t.reshape(-1).view(torch.uint8).numpy()
    raw = arr.tobytes() if copy else memoryview(arr).cast("B")
    return raw, DTYPE_TO_NAME[t.dtype], shape


def bytes_to_tensor(payload: bytes, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Rebuild a CPU tensor from raw bytes. The result owns its memory."""
    torch_dtype = NAME_TO_DTYPE[dtype]
    if len(payload) == 0:
        return torch.empty(shape, dtype=torch_dtype)
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    return buf.view(torch_dtype).reshape(shape)


def tensor_to_packet(
    tensor: torch.Tensor,
    message_type: MessageType,
    *,
    device: "DeviceAdapter | None" = None,
    lifecycle: TensorLifecycle | None = None,
    **fields,
) -> TensorPacket:
    """DETACH -> CPU_STAGING -> SERIALIZE."""
    lc = lifecycle if lifecycle is not None else TensorLifecycle()
    with lc.timed("detach"):
        t = tensor.detach()
    with lc.timed("to_cpu"):
        if device is not None:
            device.synchronize()
        t = t.to("cpu")
    with lc.timed("serialize"):
        payload, dtype, shape = tensor_to_bytes(t)
    lc.nbytes = len(payload)
    return TensorPacket(message_type=message_type, shape=shape, dtype=dtype, payload=payload, **fields)


def packet_to_tensor(
    packet: TensorPacket,
    *,
    device: "DeviceAdapter | None" = None,
    lifecycle: TensorLifecycle | None = None,
) -> torch.Tensor:
    """DESERIALIZE -> TARGET_DEVICE."""
    if packet.dtype is None:
        raise ValueError("packet carries no tensor")
    lc = lifecycle if lifecycle is not None else TensorLifecycle()
    with lc.timed("deserialize"):
        t = bytes_to_tensor(packet.payload, packet.dtype, packet.shape)
    lc.nbytes = len(packet.payload)
    if device is not None:
        with lc.timed("to_device"):
            t = device.move_tensor(t)
            device.synchronize()
    return t

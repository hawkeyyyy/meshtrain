"""TensorPacket: the unit of data-plane communication.

A packet is a small JSON-able header plus an optional raw byte payload. The
payload is always the C-contiguous bytes of a CPU tensor whose ``dtype`` and
``shape`` are described in the header. No Python objects are ever pickled.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch


class MessageType(enum.IntEnum):
    FORWARD_ACTIVATION = 1
    BACKWARD_GRADIENT = 2
    TARGET = 3
    CONTROL = 4
    ACK = 5
    ERROR = 6


# Explicit allow-list of wire dtypes. Names are stable strings, independent of
# the torch version, so a Windows CUDA worker and a macOS MPS worker agree.
DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
NAME_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in DTYPE_TO_NAME.items()}

# Scalar types allowed inside ``meta`` (no nested objects that could smuggle code).
_META_SCALARS = (str, int, float, bool, type(None))


def dtype_itemsize(name: str) -> int:
    return torch.empty((), dtype=NAME_TO_DTYPE[name]).element_size()


@dataclass
class TensorPacket:
    message_type: MessageType
    job_id: str = ""
    step_id: int = 0
    microbatch_id: int = 0
    source_worker: str = ""
    destination_worker: str = ""
    shape: tuple[int, ...] = ()
    dtype: str | None = None  # None means "no tensor payload"
    payload: bytes = b""
    tensor_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    # Small flat key/value annotations (e.g. microbatch loss, control command).
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tensor(self) -> bool:
        return self.dtype is not None

    @property
    def expected_payload_bytes(self) -> int:
        if self.dtype is None:
            return 0
        return math.prod(self.shape) * dtype_itemsize(self.dtype)

    def header(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "message_type": int(self.message_type),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "source_worker": self.source_worker,
            "destination_worker": self.destination_worker,
            "meta": self.meta,
        }

    @classmethod
    def from_header(cls, header: dict[str, Any], payload: bytes) -> "TensorPacket":
        """Build a packet from an untrusted header; raises ValueError on anything odd."""
        if not isinstance(header, dict):
            raise ValueError("packet header must be an object")
        try:
            mtype = MessageType(int(header["message_type"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid message_type: {header.get('message_type')!r}") from exc

        def _int(name: str) -> int:
            v = header.get(name, 0)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"field {name!r} must be a non-negative int")
            return v

        def _str(name: str) -> str:
            v = header.get(name, "")
            if not isinstance(v, str) or len(v) > 256:
                raise ValueError(f"field {name!r} must be a string (<=256 chars)")
            return v

        shape = header.get("shape", [])
        if not isinstance(shape, list) or len(shape) > 16 or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise ValueError(f"invalid shape: {shape!r}")
        dtype = header.get("dtype")
        if dtype is not None and dtype not in NAME_TO_DTYPE:
            raise ValueError(f"unsupported dtype: {dtype!r}")
        meta = header.get("meta", {})
        if not isinstance(meta, dict) or not all(
            isinstance(k, str) and isinstance(v, _META_SCALARS) for k, v in meta.items()
        ):
            raise ValueError("meta must be a flat object of scalars")

        packet = cls(
            message_type=mtype,
            job_id=_str("job_id"),
            step_id=_int("step_id"),
            microbatch_id=_int("microbatch_id"),
            source_worker=_str("source_worker"),
            destination_worker=_str("destination_worker"),
            shape=tuple(shape),
            dtype=dtype,
            payload=payload,
            tensor_id=_str("tensor_id"),
            meta=meta,
        )
        if len(payload) != packet.expected_payload_bytes:
            raise ValueError(
                f"payload size {len(payload)} does not match shape {tuple(shape)} "
                f"and dtype {dtype} ({packet.expected_payload_bytes} bytes)"
            )
        return packet

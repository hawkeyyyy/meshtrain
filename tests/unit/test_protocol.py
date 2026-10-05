import json

import pytest
import torch

from meshtrain.networking.protocol import (
    MAX_HEADER_BYTES,
    PREFIX,
    ProtocolError,
    decode_packet,
    encode_packet,
    parse_prefix,
)
from meshtrain.runtime.serialization import tensor_to_packet
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket


def _frame(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header).encode()
    return PREFIX.pack(b"MSHT", 1, 0, len(h), len(payload)) + h + payload


GOOD = {"tensor_id": "t", "job_id": "j", "step_id": 0, "microbatch_id": 0, "message_type": 4,
        "shape": [], "dtype": None, "source_worker": "", "destination_worker": "", "meta": {}}


def test_control_packet_without_tensor():
    p = decode_packet(encode_packet(TensorPacket(MessageType.CONTROL, meta={"command": "ping"})))
    assert p.message_type == MessageType.CONTROL and not p.has_tensor and p.meta["command"] == "ping"


def test_good_frame_parses():
    assert decode_packet(_frame(GOOD)).message_type == MessageType.CONTROL


@pytest.mark.parametrize("bad", [b"XXXX", b"MSHU"])
def test_bad_magic(bad):
    frame = bytearray(_frame(GOOD))
    frame[:4] = bad
    with pytest.raises(ProtocolError, match="magic"):
        decode_packet(bytes(frame))


def test_bad_version():
    h = json.dumps(GOOD).encode()
    with pytest.raises(ProtocolError, match="version"):
        decode_packet(PREFIX.pack(b"MSHT", 9, 0, len(h), 0) + h)


def test_oversized_header_rejected():
    with pytest.raises(ProtocolError, match="header length"):
        parse_prefix(PREFIX.pack(b"MSHT", 1, 0, MAX_HEADER_BYTES + 1, 0))


def test_oversized_payload_rejected():
    with pytest.raises(ProtocolError, match="exceeds limit"):
        parse_prefix(PREFIX.pack(b"MSHT", 1, 0, 10, 10_000), max_payload_bytes=1000)


def test_truncated_frame_rejected():
    frame = encode_packet(tensor_to_packet(torch.zeros(8), MessageType.FORWARD_ACTIVATION))
    with pytest.raises(ProtocolError):
        decode_packet(frame[:-3])


def test_non_json_header_rejected():
    h = b"\x80\x04pickle-ish"
    with pytest.raises(ProtocolError, match="malformed"):
        decode_packet(PREFIX.pack(b"MSHT", 1, 0, len(h), 0) + h)


@pytest.mark.parametrize("field,value", [
    ("message_type", 99),
    ("message_type", "FORWARD"),
    ("step_id", -1),
    ("step_id", 1.5),
    ("shape", "3,4"),
    ("shape", [-1]),
    ("dtype", "object"),
    ("meta", {"nested": {"a": 1}}),
    ("meta", [1, 2]),
    ("job_id", 5),
])
def test_invalid_header_fields_rejected(field, value):
    with pytest.raises(ProtocolError):
        decode_packet(_frame({**GOOD, field: value}))


def test_header_not_object_rejected():
    with pytest.raises(ProtocolError):
        decode_packet(_frame([1, 2, 3]))  # type: ignore[arg-type]


def test_message_type_values_are_stable():
    assert [m.name for m in MessageType] == [
        "FORWARD_ACTIVATION", "BACKWARD_GRADIENT", "TARGET", "CONTROL", "ACK", "ERROR"]

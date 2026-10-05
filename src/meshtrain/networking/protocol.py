"""Binary wire format for TensorPackets.

Frame layout (all integers big-endian):

    offset  size  field
    0       4     magic  b"MSHT"
    4       1     protocol version (1)
    5       1     reserved (0)
    6       4     header_len   (bytes of UTF-8 JSON header)
    10      8     payload_len  (bytes of raw tensor payload)
    18      H     header JSON
    18+H    P     payload

The header is plain JSON validated field-by-field by
``TensorPacket.from_header``; nothing is ever unpickled or evaluated.
"""

from __future__ import annotations

import json
import struct

from meshtrain.runtime.tensor_packet import TensorPacket

MAGIC = b"MSHT"
VERSION = 1
PREFIX = struct.Struct(">4sBBIQ")
PREFIX_SIZE = PREFIX.size  # 18

MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024**3  # 2 GiB per tensor


class ProtocolError(ValueError):
    """Raised for malformed, oversized or otherwise invalid frames."""


def encode_head(packet: TensorPacket) -> bytes:
    """Prefix + JSON header; the payload is sent separately (zero-copy)."""
    header = json.dumps(packet.header(), separators=(",", ":")).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header too large ({len(header)} bytes)")
    return PREFIX.pack(MAGIC, VERSION, 0, len(header), len(packet.payload)) + header


def encode_packet(packet: TensorPacket) -> bytes:
    return encode_head(packet) + bytes(packet.payload)


def parse_prefix(prefix: bytes, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> tuple[int, int]:
    """Validate the fixed-size prefix; return (header_len, payload_len)."""
    if len(prefix) != PREFIX_SIZE:
        raise ProtocolError("truncated frame prefix")
    magic, version, _reserved, header_len, payload_len = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"invalid header length {header_len}")
    if payload_len > max_payload_bytes:
        raise ProtocolError(f"payload of {payload_len} bytes exceeds limit {max_payload_bytes}")
    return header_len, payload_len


def parse_header(header_bytes: bytes):
    """JSON-decode a frame header (field validation happens in packet_from_header)."""
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed header: {exc}") from exc


def packet_from_header(header, payload) -> TensorPacket:
    try:
        return TensorPacket.from_header(header, payload)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc


def decode_body(header_bytes: bytes, payload: bytes) -> TensorPacket:
    return packet_from_header(parse_header(header_bytes), payload)


def decode_packet(frame: bytes, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> TensorPacket:
    header_len, payload_len = parse_prefix(frame[:PREFIX_SIZE], max_payload_bytes)
    if len(frame) != PREFIX_SIZE + header_len + payload_len:
        raise ProtocolError("frame length does not match prefix")
    header = frame[PREFIX_SIZE : PREFIX_SIZE + header_len]
    payload = frame[PREFIX_SIZE + header_len :]
    return decode_body(header, payload)

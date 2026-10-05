import pytest
import torch

from meshtrain.networking.protocol import ProtocolError, decode_packet, encode_packet
from meshtrain.runtime.serialization import bytes_to_tensor, packet_to_tensor, tensor_to_bytes, tensor_to_packet
from meshtrain.runtime.tensor_packet import DTYPE_TO_NAME, MessageType


@pytest.mark.parametrize("dtype", list(DTYPE_TO_NAME))
def test_round_trip_preserves_dtype_and_values(dtype):
    g = torch.Generator().manual_seed(0)
    if dtype == torch.bool:
        t = torch.rand(3, 5, generator=g) > 0.5
    elif dtype.is_floating_point:
        t = torch.randn(3, 5, generator=g).to(dtype)
    else:
        t = torch.randint(0, 100, (3, 5), generator=g).to(dtype)
    raw, name, shape = tensor_to_bytes(t)
    back = bytes_to_tensor(raw, name, shape)
    assert back.dtype == dtype
    assert torch.equal(back, t)


@pytest.mark.parametrize("shape", [(), (0,), (7,), (2, 3, 4), (1, 1, 1, 9), (4, 0, 2)])
def test_round_trip_preserves_shape(shape):
    t = torch.arange(int(torch.tensor(shape).prod().item()) if shape else 1, dtype=torch.float32).reshape(shape)
    back = bytes_to_tensor(*tensor_to_bytes(t))
    assert back.shape == t.shape
    assert torch.equal(back, t)


def test_non_contiguous_tensor_is_serialized_correctly():
    t = torch.randn(4, 6).t()[1:, ::2]
    assert not t.is_contiguous()
    back = bytes_to_tensor(*tensor_to_bytes(t))
    assert torch.equal(back, t)


def test_requires_grad_tensor_is_detached():
    t = torch.randn(3, requires_grad=True) * 2
    packet = tensor_to_packet(t, MessageType.FORWARD_ACTIVATION)
    back = packet_to_tensor(packet)
    assert not back.requires_grad
    assert torch.equal(back, t.detach())


def test_packet_round_trip_through_wire_format():
    t = torch.randn(2, 3, dtype=torch.float64)
    packet = tensor_to_packet(t, MessageType.BACKWARD_GRADIENT, job_id="j", step_id=4, microbatch_id=2,
                              source_worker="a", destination_worker="b", meta={"loss": 1.5})
    decoded = decode_packet(encode_packet(packet))
    assert decoded.message_type == MessageType.BACKWARD_GRADIENT
    assert (decoded.job_id, decoded.step_id, decoded.microbatch_id) == ("j", 4, 2)
    assert (decoded.source_worker, decoded.destination_worker) == ("a", "b")
    assert decoded.meta == {"loss": 1.5}
    assert decoded.tensor_id == packet.tensor_id
    assert torch.equal(packet_to_tensor(decoded), t)


def test_received_tensor_is_writable_and_owns_memory():
    t = torch.randn(5)
    back = bytes_to_tensor(*tensor_to_bytes(t))
    back.add_(1)  # must not raise (frombuffer on bytes would be read-only)
    assert torch.allclose(back, t + 1)


def test_non_cpu_tensor_must_be_staged():
    with pytest.raises(ValueError):
        tensor_to_bytes(torch.empty(2, device="meta"))


def test_unsupported_dtype_rejected():
    with pytest.raises(ValueError):
        tensor_to_bytes(torch.zeros(2, dtype=torch.complex64))


def test_payload_size_mismatch_rejected():
    packet = tensor_to_packet(torch.zeros(4), MessageType.FORWARD_ACTIVATION)
    frame = bytearray(encode_packet(packet))
    # Drop the last 4 bytes of payload and fix up the prefix length.
    import struct
    from meshtrain.networking.protocol import PREFIX
    magic, ver, res, hl, pl = PREFIX.unpack(bytes(frame[: PREFIX.size]))
    frame = PREFIX.pack(magic, ver, res, hl, pl - 4) + bytes(frame[PREFIX.size: -4])
    with pytest.raises(ProtocolError, match="payload size"):
        decode_packet(frame)

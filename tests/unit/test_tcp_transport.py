import socket
import threading
import time

import pytest
import torch

from meshtrain.networking.protocol import ProtocolError
from meshtrain.networking.tcp import TCPListener, connect
from meshtrain.networking.transport import TransportClosed, TransportTimeout
from meshtrain.runtime.tensor_packet import MessageType


@pytest.fixture
def pair():
    listener = TCPListener("127.0.0.1", 0, token="secret")
    box = {}
    th = threading.Thread(target=lambda: box.update(server=listener.accept(timeout=5)))
    th.start()
    client = connect("127.0.0.1", listener.port, timeout=5, hello={"job_id": "j", "stage": 0, "token": "secret"})
    th.join()
    server, hello = box["server"]
    assert hello["stage"] == 0 and hello["job_id"] == "j"
    yield client, server
    client.close()
    server.close()
    listener.close()


def test_tensor_round_trip_over_tcp(pair):
    client, server = pair
    t = torch.randn(64, 512)
    client.send_tensor(t, MessageType.FORWARD_ACTIVATION, job_id="j", step_id=3, microbatch_id=1)
    back, packet = server.recv_tensor(timeout=5)
    assert torch.equal(back, t)
    assert (packet.step_id, packet.microbatch_id) == (3, 1)
    # and the reverse direction on the same connection
    server.send_tensor(t * 2, MessageType.BACKWARD_GRADIENT, job_id="j")
    back2, _ = client.recv_tensor(timeout=5)
    assert torch.equal(back2, t * 2)


def test_large_tensor_over_tcp(pair):
    client, server = pair
    t = torch.randn(4, 1024, 1024)  # 16 MB
    th = threading.Thread(target=client.send_tensor, args=(t, MessageType.FORWARD_ACTIVATION))
    th.start()
    back, _ = server.recv_tensor(timeout=10)
    th.join()
    assert torch.equal(back, t)
    assert client.bytes_sent >= 16 * 1024 * 1024


def test_recv_timeout(pair):
    _, server = pair
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout):
        server.recv_packet(timeout=0.3)
    assert time.monotonic() - t0 < 2


def test_peer_disconnect_detected(pair):
    client, server = pair
    client.close()
    with pytest.raises(TransportClosed):
        server.recv_packet(timeout=2)


def test_connect_timeout_when_nobody_listens():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(TransportTimeout):
        connect("127.0.0.1", port, timeout=0.5)


def test_bad_token_rejected():
    listener = TCPListener("127.0.0.1", 0, token="secret")
    errors = []

    def serve():
        try:
            listener.accept(timeout=5)
        except Exception as exc:
            errors.append(exc)

    th = threading.Thread(target=serve)
    th.start()
    c = connect("127.0.0.1", listener.port, timeout=5, hello={"token": "wrong"})
    th.join()
    assert errors and isinstance(errors[0], ProtocolError)
    c.close()
    listener.close()


def test_garbage_bytes_rejected(pair):
    client, server = pair
    client.sock.sendall(b"GET / HTTP/1.1\r\n\r\n" + b"x" * 10)
    with pytest.raises(ProtocolError):
        server.recv_packet(timeout=2)


def test_oversized_payload_rejected_by_receiver(pair):
    client, server = pair
    server.max_payload_bytes = 1000
    client.send_tensor(torch.zeros(1000), MessageType.FORWARD_ACTIVATION)  # 4000 bytes
    with pytest.raises(ProtocolError, match="exceeds limit"):
        server.recv_packet(timeout=2)

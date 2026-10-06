"""V2.5 remote tensor service: protocol, versions, checksums, reservations, failures."""

import threading
import time

import pytest
import torch

from meshtrain.networking.tensor_server import (
    RemoteMemoryError,
    RemoteTensorClient,
    TensorServer,
    VersionConflict,
    start_local_server,
)
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket

MB = 1024**2


@pytest.fixture
def server():
    srv, port, stop = start_local_server(64 * MB, token="tok", lease_s=60)
    yield srv, port
    stop.set()


def _client(port, job="job1", **kw):
    return RemoteTensorClient("127.0.0.1", port, token="tok", job_id=job, timeout_s=10, **kw)


def test_put_get_metadata_delete_round_trip(server):
    srv, port = server
    c = _client(port, checksum="crc32")
    c.reserve(8 * MB)
    t = torch.randn(256, 128)
    assert c.put("model.layers.3.qkv.weight", t, old_version=-1, new_version=1) == 1
    out = torch.empty(256, 128)
    assert c.get("model.layers.3.qkv.weight", out, version=1) == 1
    assert torch.equal(out, t)
    md = c.metadata("model.layers.3.qkv.weight")
    assert md["version"] == 1 and md["bytes"] == t.numel() * 4 and md["tensor_dtype"] == "float32" and md["tensor_shape"] == "256,128"
    assert c.exists("model.layers.3.qkv.weight") and not c.exists("nope")
    assert srv.used == t.numel() * 4
    c.delete("model.layers.3.qkv.weight")
    assert srv.used == 0 and not c.exists("model.layers.3.qkv.weight")
    c.release()
    assert srv.reserved == 0
    c.close()


def test_version_protocol_rejects_stale_writers(server):
    _, port = server
    a, b = _client(port), _client(port)
    a.reserve(MB)
    a.put("w", torch.zeros(4), old_version=-1, new_version=1)
    a.put("w", torch.ones(4), old_version=1, new_version=2)
    with pytest.raises(VersionConflict) as e:          # b still believes version 1 is current
        b.put("w", torch.full((4,), 7.0), old_version=1, new_version=2)
    assert e.value.code == "VERSION_CONFLICT"
    out = torch.empty(4)
    a.get("w", out)
    assert torch.equal(out, torch.ones(4))             # the stale write changed nothing
    with pytest.raises(RemoteMemoryError, match="VERSION|version"):
        a.get("w", out, version=1)
    with pytest.raises(VersionConflict):
        a.put("w", torch.ones(4), old_version=-1, new_version=1)   # "create" of an existing tensor
    with pytest.raises(RemoteMemoryError, match="must exceed"):
        a.put("w", torch.ones(4), old_version=2, new_version=2)


def test_checksum_mismatch_is_rejected():
    srv = TensorServer(MB)
    srv.process(TensorPacket(MessageType.CONTROL, meta={"op": "RESERVE", "job": "j", "bytes": 1024}))
    payload = torch.arange(4, dtype=torch.float32).numpy().tobytes()
    bad = TensorPacket(MessageType.CONTROL, payload=payload, dtype="float32", shape=(4,),
                       meta={"op": "PUT", "job": "j", "tensor_id": "t", "old_version": -1, "new_version": 1,
                             "crc32": 12345})
    reply = srv.process(bad)
    assert reply.message_type == MessageType.ERROR and reply.meta["code"] == "CHECKSUM"
    assert srv.used == 0


def test_reservation_accounting_and_exhaustion(server):
    srv, port = server
    a, b = _client(port, "jobA"), _client(port, "jobB")
    a.reserve(40 * MB)
    with pytest.raises(RemoteMemoryError) as e:
        b.reserve(30 * MB)                          # 40 + 30 > 64 MB budget
    assert e.value.code == "INSUFFICIENT_MEMORY"
    b.reserve(20 * MB)
    with pytest.raises(RemoteMemoryError) as e:     # a job cannot store beyond its own reservation
        b.put("big", torch.zeros(6 * MB), old_version=-1, new_version=1)   # 24 MB
    assert e.value.code == "RESERVATION_EXCEEDED"
    with pytest.raises(RemoteMemoryError) as e:
        _client(port, "jobC").put("x", torch.zeros(1), old_version=-1, new_version=1)
    assert e.value.code == "NO_RESERVATION"
    a.release()
    b.reserve(50 * MB)                              # room again after jobA released
    st = b.stats()
    assert st["budget"] == 64 * MB and st["reserved"] == 50 * MB


def test_bad_token_and_bad_requests_rejected(server):
    _, port = server
    bad = RemoteTensorClient("127.0.0.1", port, token="wrong", job_id="j", timeout_s=3)
    with pytest.raises(RemoteMemoryError):
        bad.reserve(1)
    c = _client(port)
    with pytest.raises(RemoteMemoryError, match="BAD_REQUEST|bad request|unknown op"):
        c.request("FORMAT_DISK")
    c.reserve(MB)
    with pytest.raises(RemoteMemoryError, match="too long"):
        c.put("x" * 500, torch.zeros(1), old_version=-1, new_version=1)


def test_lease_expiry_frees_a_silent_jobs_memory():
    srv, port, stop = start_local_server(16 * MB, lease_s=0.6)
    try:
        c = RemoteTensorClient("127.0.0.1", port, token=None, job_id="crashed", timeout_s=5)
        c.reserve(8 * MB)
        c.put("w", torch.zeros(1024), old_version=-1, new_version=1)
        c.close()                                   # client "crashes": no RELEASE, no heartbeat
        deadline = time.time() + 10
        while srv.reserved and time.time() < deadline:
            time.sleep(0.2)
        assert srv.reserved == 0 and srv.used == 0
    finally:
        stop.set()


def test_disconnect_and_timeout_fail_cleanly():
    srv, port, stop = start_local_server(16 * MB)
    c = RemoteTensorClient("127.0.0.1", port, token=None, job_id="j", timeout_s=2)
    c.reserve(MB)
    stop.set()                                      # remote memory worker disappears
    srv.close()
    time.sleep(0.5)
    t0 = time.time()
    with pytest.raises(RemoteMemoryError) as e:
        for _ in range(5):
            c.put("w", torch.zeros(16), old_version=-1, new_version=1)
            time.sleep(0.2)
    assert e.value.code in ("DISCONNECTED", "UNREACHABLE", "TIMEOUT")
    assert time.time() - t0 < 60                    # no hang


def test_inflight_limit_bounds_concurrent_requests(server):
    _, port = server
    c = _client(port, max_inflight_fetches=2)
    c.reserve(32 * MB)
    c.put("w", torch.zeros(1024 * 1024), old_version=-1, new_version=1)
    seen = []
    lock = threading.Lock()

    def fetch():
        out = torch.empty(1024 * 1024)
        c.get("w", out)
        with lock:
            seen.append(c._made["fetch"])

    ts = [threading.Thread(target=fetch) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(seen) == 6 and max(seen) <= 2        # never more than 2 fetch connections


def test_probe_reports_rtt_and_bandwidth(server):
    _, port = server
    r = _client(port).probe(pings=3, payload_mb=4)
    assert r["rtt_s"] > 0 and r["bandwidth_Bps"] > 0

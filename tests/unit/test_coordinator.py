import time

import pytest
from fastapi.testclient import TestClient

from meshtrain.coordinator.server import Coordinator, create_app

HW = {"hostname": "h", "platform": "linux", "cpu_count": 4, "ram_total": 8 * 1024**3,
      "ram_available": 6 * 1024**3, "accelerators": [], "capabilities": ["cpu"]}
REG = {"name": "w1", "hardware": HW, "backend": "cpu", "device": {"backend": "cpu", "name": "x",
       "memory_total": 8 * 1024**3}, "data_host": "127.0.0.1", "data_port": 29501}


@pytest.fixture
def env():
    c = Coordinator("tok", runs_dir="/tmp/meshtrain-test-runs", heartbeat_timeout_s=0.6, verbose=False)
    client = TestClient(create_app(c), headers={"X-MeshTrain-Token": "tok"})
    yield c, client
    c.shutdown()


def test_registration_requires_token(env):
    c, client = env
    r = client.post("/workers/register", json=REG, headers={"X-MeshTrain-Token": "wrong"})
    assert r.status_code == 401
    r = TestClient(create_app(c)).post("/workers/register", json=REG)
    assert r.status_code == 401


def test_register_and_status(env):
    c, client = env
    wid = client.post("/workers/register", json=REG).json()["worker_id"]
    assert wid == "w1"
    st = client.get("/cluster/status").json()
    assert st["workers"][0]["status"] == "ONLINE" and st["workers"][0]["backend"] == "cpu"
    # second worker with the same name but a different address gets a distinct id
    wid2 = client.post("/workers/register", json={**REG, "data_port": 29502}).json()["worker_id"]
    assert wid2 != wid


def test_heartbeat_keeps_worker_online_and_silence_marks_offline(env):
    c, client = env
    wid = client.post("/workers/register", json=REG).json()["worker_id"]
    for _ in range(4):
        time.sleep(0.3)
        assert client.post(f"/workers/{wid}/heartbeat", json={"memory": {"allocated": 1}}).status_code == 200
    assert c.registry.get(wid).status.value == "ONLINE"
    time.sleep(1.5)
    assert c.registry.get(wid).status.value == "OFFLINE"
    # a heartbeat brings it back
    client.post(f"/workers/{wid}/heartbeat", json={})
    assert c.registry.get(wid).status.value == "ONLINE"


def test_unknown_worker_heartbeat_is_404(env):
    _, client = env
    assert client.post("/workers/nope/heartbeat", json={}).status_code == 404


def test_command_long_poll(env):
    c, client = env
    wid = client.post("/workers/register", json=REG).json()["worker_id"]
    t0 = time.time()
    assert client.get(f"/workers/{wid}/commands", params={"timeout": 0.3}).json()["commands"] == []
    assert time.time() - t0 >= 0.25
    c.registry.send(wid, {"type": "STOP_JOB", "job_id": "j"})
    assert client.get(f"/workers/{wid}/commands", params={"timeout": 1}).json()["commands"][0]["type"] == "STOP_JOB"


def test_invalid_config_rejected(env):
    _, client = env
    client.post("/workers/register", json=REG)
    r = client.post("/plan", json={"config": {"model": {"type": "resnet"}}})
    assert r.status_code == 400 and "invalid" in r.json()["detail"]


def test_plan_dry_run(env):
    _, client = env
    client.post("/workers/register", json=REG)
    client.post("/workers/register", json={**REG, "name": "w2", "data_port": 29502})
    r = client.post("/plan", json={"config": {"model": {"type": "mlp"}, "placement": {"strategy": "auto",
                                                                                     "num_stages": 2}}})
    assert r.status_code == 200, r.text
    assert r.json()["plan"]["feasible"] and "Stage 1" in r.json()["text"]


def test_oversized_request_rejected(env):
    _, client = env
    r = client.post("/workers/register", content=b"x" * (9 * 1024 * 1024),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413

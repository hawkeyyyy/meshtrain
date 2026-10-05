"""End-to-end: real coordinator (HTTP) + worker processes + TCP data plane on one host."""

import os
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn

from meshtrain.coordinator.server import Coordinator, create_app
from meshtrain.networking.control import ControlClient

TOKEN = "test-token"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cluster(tmp_path):
    port = _free_port()
    coord = Coordinator(TOKEN, runs_dir=str(tmp_path / "runs"), heartbeat_timeout_s=3.0, verbose=False)
    server = uvicorn.Server(uvicorn.Config(create_app(coord), host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    client = ControlClient(f"127.0.0.1:{port}", TOKEN)
    for _ in range(100):
        try:
            client.status()
            break
        except Exception:
            time.sleep(0.1)
    procs = []

    def start_worker(name, extra_env=None):
        env = {**os.environ, "MESHTRAIN_TOKEN": TOKEN, "OMP_NUM_THREADS": "1", **(extra_env or {})}
        p = subprocess.Popen([sys.executable, "-m", "meshtrain.cli", "worker", "join", f"127.0.0.1:{port}",
                              "--device", "cpu", "--name", name, "--data-port", "0", "--advertise-host",
                              "127.0.0.1", "--quick-benchmark", "--runs-dir", str(tmp_path / "worker-runs")],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p)
        return p

    def wait_online(n, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ws = [w for w in client.status()["workers"] if w["status"] == "ONLINE"]
            if len(ws) >= n:
                return ws
            time.sleep(0.2)
        raise TimeoutError(f"only {len(ws)} workers online")

    yield client, coord, start_worker, wait_online
    for p in procs:
        p.kill()
        p.wait()
    server.should_exit = True
    th.join(5)
    coord.shutdown()


def _wait_job(client, job_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.job(job_id)
        if j["status"] != "RUNNING":
            return j
        time.sleep(0.3)
    raise TimeoutError("job did not finish")


CFG = {
    "job": {"name": "e2e"},
    "model": {"type": "mlp"},
    "training": {"batch_size": 64, "microbatch_size": 16, "learning_rate": 0.001, "optimizer": "adam",
                 "steps": 40, "log_every": 10},
    "placement": {"strategy": "auto", "num_stages": 3},
    "workers": {"allow": ["cpu"]},
    "network": {"timeout_s": 20, "connect_timeout_s": 20},
}


def test_three_worker_cluster_benchmark_and_training(cluster):
    client, coord, start_worker, wait_online = cluster
    for n in ("alpha", "beta", "gamma"):
        start_worker(n)
    wait_online(3)

    client.start_benchmark(pings=3, payload_mb=2)
    t0 = time.time()
    while client.benchmark()["status"] == "RUNNING" and time.time() - t0 < 120:
        time.sleep(0.5)
    b = client.benchmark()
    assert b["status"] == "DONE" and not b["errors"], b.get("errors")
    assert len(b["workers"]) == 3 and len(b["links"]) == 6  # directional i->j
    assert max(r["compute_score"] for r in b["workers"].values()) == 1.0

    from meshtrain.planner.report import format_placement

    plan = client.plan(CFG)["plan"]
    assert "Boundary 1 -> 2" in format_placement(plan)
    job = client.start_job(CFG)
    assert len(job["plan"]["stages"]) == 3
    j = _wait_job(client, job["job_id"])
    assert j["status"] == "COMPLETED", j["error"]
    losses = [l for _, l in j["losses"]]
    assert len(losses) == 40 and losses[-1] < losses[0]
    s = j["summary"]
    assert set(map(int, s["stages"])) == {0, 1, 2}
    assert s["bytes_per_step"] > 0
    assert os.path.exists(os.path.join(j["run_dir"], "metrics.jsonl"))
    acc = s["prediction_accuracy"]
    assert acc["step"]["actual_s"] > 0 and set(map(int, acc["stages"])) == {0, 1, 2}
    assert os.path.exists(os.path.join(j["run_dir"], "prediction_accuracy.json"))
    # workers are free again
    assert all(w["status"] == "ONLINE" for w in client.status()["workers"])


def test_worker_death_fails_job_cleanly(cluster):
    client, coord, start_worker, wait_online = cluster
    procs = [start_worker(n) for n in ("a1", "a2")]
    wait_online(2)
    cfg = {**CFG, "training": {**CFG["training"], "steps": 100000}, "placement": {"strategy": "equal",
                                                                                   "num_stages": 2}}
    job = client.start_job(cfg)
    time.sleep(3)
    victim = job["plan"]["stages"][1]["worker_id"]
    names = {"a1": procs[0], "a2": procs[1]}
    names[victim].kill()
    t0 = time.time()
    j = _wait_job(client, job["job_id"], timeout=60)
    elapsed = time.time() - t0
    assert j["status"] == "FAILED"
    assert victim in j["error"] or "disconnected" in j["error"], j["error"]
    assert elapsed < 30, "failure must be detected quickly"
    survivor = [w for w in client.status()["workers"] if w["worker_id"] != victim][0]
    t0 = time.time()
    while survivor["status"] != "ONLINE" and time.time() - t0 < 10:
        time.sleep(0.2)
        survivor = [w for w in client.status()["workers"] if w["worker_id"] != victim][0]
    assert survivor["status"] == "ONLINE"


def test_startup_memory_failure_triggers_replan(cluster):
    """The planner believes both CPU workers have plenty of RAM; 'small' really has 0.12 GB.
    Startup validation must catch it before training, and the coordinator must replan
    with the measured capacity instead of failing or crashing mid-step."""
    client, coord, start_worker, wait_online = cluster
    start_worker("big")
    start_worker("small", {"MESHTRAIN_EMULATE_DEVICE_MEMORY_GB": "0.12"})
    wait_online(2)
    cfg = {
        "job": {"name": "replan"},
        "model": {"type": "mlp", "sizes": [256, 2048, 2048, 2048, 2048, 2048, 10]},
        "training": {"batch_size": 16, "microbatch_size": 4, "learning_rate": 0.001, "optimizer": "adamw",
                     "steps": 3, "log_every": 1},
        "placement": {"strategy": "auto", "num_stages": 2},
        "workers": {"allow": ["cpu"]},
        "network": {"timeout_s": 20, "connect_timeout_s": 20},
        "memory": {"probe_allocation": False},
    }
    job = client.start_job(cfg)
    j = _wait_job(client, job["job_id"], timeout=180)
    assert j["status"] == "COMPLETED", j["error"]
    assert j["attempt"] >= 1, "expected at least one startup replan"
    first = j["attempts"][0]
    assert first["failure"]["worker"] == "small"
    assert "startup" in first["failure"]["error"]
    small_layers = [p["layers"] for p in j["attempts"][-1]["placement"] if p["worker"] == "small"]
    first_small = [p["layers"] for p in first["placement"] if p["worker"] == "small"]
    size = lambda r: r[0][1] - r[0][0]  # noqa: E731
    assert small_layers, "the replanned placement still uses the small worker, with less of the model"
    assert size(small_layers) < size(first_small)
    assert j["memory_reports"], "startup memory validation reports are recorded"

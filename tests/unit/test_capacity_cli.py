import pytest

from meshtrain import cli
from meshtrain.experiments import capacity


@pytest.mark.parametrize("source", ["saved", "environment", "flags_before", "flags_after"])
def test_hardware_capacity_uses_cluster_connection_precedence(tmp_path, monkeypatch, source):
    monkeypatch.setenv("MESHTRAIN_HOME", str(tmp_path))
    monkeypatch.delenv("MESHTRAIN_COORDINATOR", raising=False)
    monkeypatch.delenv("MESHTRAIN_TOKEN", raising=False)
    monkeypatch.delenv("MESHTRAIN_QUIET", raising=False)
    cli.save_cluster("saved-cluster:8090", "saved-token")
    expected = ("http://saved-cluster:8090", "saved-token")
    if source != "saved":
        monkeypatch.setenv("MESHTRAIN_COORDINATOR", "env-cluster:8091")
        monkeypatch.setenv("MESHTRAIN_TOKEN", "env-token")
        expected = ("http://env-cluster:8091", "env-token")
    command = ["experiment", "capacity", "--mode", "hardware", "--steps", "3"]
    if source.startswith("flags"):
        flags = ["--coordinator", "flag-cluster:8092", "--token", "flag-token"]
        command = flags + command if source == "flags_before" else command + flags
        expected = ("http://flag-cluster:8092", "flag-token")

    def run_hardware(client, *, steps):
        try:
            assert (client.base, client.http.headers["X-MeshTrain-Token"]) == expected
            assert steps == 3
            return "capacity report"
        finally:
            client.http.close()

    monkeypatch.setattr(capacity, "run_hardware", run_hardware)
    assert cli.main(command) == 0


def test_direct_capacity_call_uses_remembered_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHTRAIN_HOME", str(tmp_path))
    monkeypatch.delenv("MESHTRAIN_COORDINATOR", raising=False)
    monkeypatch.delenv("MESHTRAIN_TOKEN", raising=False)
    cli.save_cluster("saved-cluster:8090", "saved-token")

    def run_hardware(client, *, steps):
        try:
            assert client.base == "http://saved-cluster:8090"
            assert client.http.headers["X-MeshTrain-Token"] == "saved-token"
            return "capacity report"
        finally:
            client.http.close()

    monkeypatch.setattr(capacity, "run_hardware", run_hardware)
    assert capacity.run_experiment5(mode="hardware") == "capacity report"


def test_emulated_capacity_does_not_resolve_a_cluster(monkeypatch):
    def no_client(*args):
        raise AssertionError("emulated capacity should not connect to a coordinator")

    monkeypatch.setattr(cli, "_client", no_client)
    monkeypatch.setattr(capacity, "run_emulated", lambda *, steps: f"emulated {steps}")
    assert cli.main(["experiment", "capacity", "--steps", "3"]) == 0


def test_hardware_capacity_mesh_uses_all_workers_and_reports_stage_count():
    class Client:
        configs = []

        def status(self):
            return {"workers": [{"worker_id": w, "status": "ONLINE"} for w in ("laptop", "server")]}

        def start_job(self, cfg):
            self.configs.append(cfg)
            return {"job_id": len(self.configs) - 1}

        def job(self, job_id):
            cfg = self.configs[job_id]
            n = cfg["placement"].get("num_stages") or 1
            return {"status": "COMPLETED", "summary": {"steps_per_s": 1},
                    "stage_workers": list(range(n))}

    client = Client()
    report = capacity.run_hardware(client, sizes=[(2, 128)], steps=3, write_doc=False)
    assert len(client.configs) == 3
    mesh = client.configs[-1]
    assert mesh["placement"]["num_stages"] == 2
    assert mesh["placement"]["enforce_memory_check"]
    assert "stages at P_max" in report and "one per online worker" in report


def test_hardware_capacity_does_not_run_a_mesh_comparison_with_the_server_offline():
    class Client:
        def status(self):
            return {"workers": [{"worker_id": "laptop", "status": "ONLINE"},
                                {"worker_id": "server", "status": "OFFLINE"}]}

        def start_job(self, cfg):
            raise AssertionError("a single-worker cluster must not start a capacity comparison")

    with pytest.raises(ValueError, match="at least two online workers"):
        capacity.run_hardware(Client(), write_doc=False)

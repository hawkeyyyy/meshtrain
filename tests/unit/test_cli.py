from meshtrain.cli import build_parser, format_status

GB = 1024**3


def test_global_flags_before_or_after_subcommand():
    p = build_parser()
    a = p.parse_args(["--coordinator", "h:1", "--token", "t", "cluster", "status"])
    b = p.parse_args(["cluster", "status", "--coordinator", "h:1", "--token", "t"])
    assert (a.coordinator, a.token) == (b.coordinator, b.token) == ("h:1", "t")
    c = p.parse_args(["worker", "join", "192.168.1.20:8080", "--device", "mps"])
    assert c.coordinator_address == "192.168.1.20:8080" and c.device == "mps" and c.token is None


def test_status_table_marks_unified_memory():
    status = {"workers": [
        {"worker_id": "samyog-pc", "backend": "cuda", "device": {"name": "RTX 4060", "memory_total": 8 * GB},
         "ram_total": 32 * GB, "status": "ONLINE", "benchmark": {"compute_score": 0.72}},
        {"worker_id": "m2-air", "backend": "mps", "device": {"name": "Apple arm64", "memory_total": 16 * GB,
                                                             "unified_memory": True},
         "ram_total": 16 * GB, "status": "ONLINE"},
        {"worker_id": "old-pc", "backend": "cpu", "device": {}, "ram_total": 8 * GB, "status": "OFFLINE"},
    ], "jobs": []}
    text = format_status(status)
    assert "16.0 GB*" in text and "* unified memory" in text and "OFFLINE" in text and "0.72" in text


def test_parse_join_target():
    from meshtrain.cli import parse_join_target

    assert parse_join_target("abc123@192.168.1.20") == ("192.168.1.20:8080", "abc123")
    assert parse_join_target("abc123@192.168.1.20:9000") == ("192.168.1.20:9000", "abc123")
    assert parse_join_target("192.168.1.20:8080") == ("192.168.1.20:8080", None)
    assert parse_join_target("http://host") == ("host:8080", None)


def test_cluster_is_remembered(tmp_path, monkeypatch):
    from meshtrain import cli

    monkeypatch.setenv("MESHTRAIN_HOME", str(tmp_path))
    monkeypatch.delenv("MESHTRAIN_TOKEN", raising=False)
    monkeypatch.delenv("MESHTRAIN_COORDINATOR", raising=False)
    assert cli.load_saved() == {}
    cli.save_cluster("10.0.0.5:8080", "tok")
    args = build_parser().parse_args(["status"])
    client = cli._client(args)
    assert client.base == "http://10.0.0.5:8080"
    assert client.http.headers["X-MeshTrain-Token"] == "tok"


def test_short_commands_parse():
    p = build_parser()
    a = p.parse_args(["join", "k3f9x2@192.168.1.20", "--device", "mps"])
    assert a.coordinator_address == "k3f9x2@192.168.1.20" and a.device == "mps"
    assert p.parse_args(["start", "--port", "9000"]).port == 9000
    b = p.parse_args(["benchmark"])
    assert b.func.__name__ == "cmd_benchmark" and b.what == "cluster"
    bp = p.parse_args(["benchmark", "pipeline", "--bandwidth-mbps", "50", "--devices", "cuda,cuda"])
    assert bp.what == "pipeline" and bp.bandwidth_mbps == 50 and bp.devices == "cuda,cuda"
    assert p.parse_args(["cluster", "benchmark"]).func.__name__ == "cmd_cluster_benchmark"
    assert p.parse_args(["inspect", "placement", "c.yaml"]).config == "c.yaml"


def test_capacity_hardware_uses_remembered_cluster(tmp_path, monkeypatch):
    from meshtrain import cli
    from meshtrain.experiments import capacity

    monkeypatch.setenv("MESHTRAIN_HOME", str(tmp_path))
    monkeypatch.delenv("MESHTRAIN_TOKEN", raising=False)
    monkeypatch.delenv("MESHTRAIN_COORDINATOR", raising=False)
    cli.save_cluster("10.0.0.5:8080", "secret")
    seen = {}
    monkeypatch.setattr(capacity, "run_hardware", lambda client, steps: seen.update(
        base=client.base, token=client.http.headers["X-MeshTrain-Token"]) or "ok")
    assert cli.main(["experiment", "capacity", "--mode", "hardware"]) == 0
    assert seen == {"base": "http://10.0.0.5:8080", "token": "secret"}

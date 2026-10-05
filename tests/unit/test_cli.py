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

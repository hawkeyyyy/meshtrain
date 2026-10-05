import io
import json
from pathlib import Path

import pytest

from meshtrain.config import ConfigError, load_config, parse_config
from meshtrain.experiments.results_doc import update_section
from meshtrain.runtime.serialization import TensorLifecycle, packet_to_tensor, tensor_to_packet
from meshtrain.runtime.tensor_packet import MessageType
from meshtrain.telemetry import EventLogger

CONFIGS = sorted(Path(__file__).resolve().parents[2].joinpath("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    cfg = load_config(path)
    assert cfg.training.batch_size % cfg.training.num_microbatches == 0
    cfg.build_model_spec()


@pytest.mark.parametrize("bad", [
    {"model": {"type": "llama"}},
    {"model": {"type": "mlp"}, "training": {"batch_size": 10, "microbatch_size": 3}},
    {"model": {"type": "mlp"}, "training": {"learning_rate": -1}},
    {"model": {"type": "tiny_transformer", "hidden_size": 100, "heads": 3}},
    {"model": {"type": "mlp"}, "placement": {"strategy": "manual"}},
    {"model": {"type": "mlp"}, "placement": {"strategy": "manual", "stages": [{"layers": [0, 3]}, {"layers": [4, 7]}]}},
    {"model": {"type": "mlp"}, "placement": {"strategy": "manual", "stages": [{"layers": [0, 3]}]}},
    {"model": {"type": "mlp"}, "unknown_section": {}},
    {"model": {"type": "mlp"}, "workers": {"allow": ["tpu"]}},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_event_log_line_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("MESHTRAIN_QUIET", raising=False)
    buf = io.StringIO()
    log = EventLogger("rtx8", "job1", stream=buf, jsonl_path=tmp_path / "e.jsonl")
    log.log("FORWARD_COMPLETE", 12, 3, output=16_200_000, compute_s=0.083)
    line = buf.getvalue()
    for part in ("worker=rtx8", "job=job1", "step=12", "mb=3", "FORWARD_COMPLETE", "output=16.2MB", "compute_s=83.0ms"):
        assert part in line
    rec = json.loads((tmp_path / "e.jsonl").read_text())
    assert {"timestamp", "worker", "job", "step", "microbatch", "event"} <= rec.keys()


def test_tensor_lifecycle_phases_timed():
    import torch

    lc = TensorLifecycle()
    p = tensor_to_packet(torch.randn(256, 256), MessageType.FORWARD_ACTIVATION, lifecycle=lc)
    lc2 = TensorLifecycle()
    packet_to_tensor(p, lifecycle=lc2)
    assert {"detach", "to_cpu", "serialize"} <= lc.timings.keys() and lc.nbytes == 256 * 256 * 4
    assert "deserialize" in lc2.timings


def test_results_doc_section_update(tmp_path):
    doc = tmp_path / "r.md"
    doc.write_text("# R\n\n<!-- BEGIN:a -->\nold\n<!-- END:a -->\n\ntail\n")
    update_section("a", "new $1 \\1 content", doc)
    update_section("b", "added", doc)
    text = doc.read_text()
    assert "new $1 \\1 content" in text and "old" not in text and "tail" in text and "<!-- BEGIN:b -->" in text

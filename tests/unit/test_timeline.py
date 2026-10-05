import json

import pytest

from meshtrain.runtime.timeline import Timeline, _intersect, _length, _union, write_chrome_trace


def test_interval_arithmetic():
    u = _union([(0, 2), (1, 3), (5, 6), (6, 7)])
    assert u == [(0, 3), (5, 7)] and _length(u) == 5
    assert _intersect([(0, 3), (5, 7)], [(2, 6)]) == [(2, 3), (5, 6)]


def test_step_metrics_overlap_and_exposure():
    tl = Timeline("w", 0)
    tl.add("FORWARD_COMPUTE", 0.0, 4.0, 0, 0)
    tl.add("NETWORK_SEND", 2.0, 6.0, 0, 0)     # 2s hidden under compute, 2s exposed
    tl.add("BACKWARD_COMPUTE", 7.0, 9.0, 0, 0)
    m = tl.step_metrics(0, (0.0, 10.0))
    assert m["compute_s"] == pytest.approx(6.0)
    assert m["communication_s"] == pytest.approx(4.0)
    assert m["overlapped_s"] == pytest.approx(2.0)
    assert m["exposed_communication_s"] == pytest.approx(2.0)
    assert m["overlap_ratio"] == pytest.approx(0.5)
    assert m["idle_s"] == pytest.approx(2.0)  # [6,7) and [9,10)


def test_queue_wait_is_not_communication_work():
    tl = Timeline("w", 0)
    tl.add("QUEUE_WAIT", 0.0, 5.0, 0, 0)
    assert tl.step_metrics(0, (0.0, 5.0))["communication_s"] == 0.0


def test_chrome_trace_export(tmp_path):
    tl = Timeline("rtx", 1)
    with tl.span("FORWARD_COMPUTE", 3, 2):
        pass
    path = write_chrome_trace(tmp_path / "timeline.json", [tl.to_chrome_events()])
    data = json.loads(path.read_text())
    xs = [e for e in data["traceEvents"] if e["ph"] == "X"]
    assert xs[0]["name"] == "FORWARD_COMPUTE mb2" and xs[0]["args"]["step"] == 3 and xs[0]["pid"] == 1
    assert any(e["ph"] == "M" and e["name"] == "process_name" for e in data["traceEvents"])

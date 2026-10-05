import json

from meshtrain.runtime.trace_report import ascii_timeline, format_breakdown, worker_breakdown, write_run_timeline


def _spans():
    return [
        {"category": "FORWARD_COMPUTE", "start": 0.0, "end": 0.4, "thread": "MainThread", "step": 1, "microbatch": 0,
         "args": None},
        {"category": "NETWORK_SEND", "start": 0.3, "end": 0.6, "thread": "send-downstream", "step": 1,
         "microbatch": 0, "args": {"bytes": 10}},
        {"category": "BACKWARD_COMPUTE", "start": 0.7, "end": 1.0, "thread": "MainThread", "step": 1,
         "microbatch": 0, "args": None},
    ]


def test_merged_chrome_trace(tmp_path):
    path = write_run_timeline(tmp_path / "timeline.json", [(0, "rtx", _spans()), (1, "m2", _spans())])
    ev = json.loads(path.read_text())["traceEvents"]
    assert {e["pid"] for e in ev if e["ph"] == "X"} == {0, 1}
    assert any(e["name"] == "thread_name" and e["args"]["name"] == "send-downstream" for e in ev)


def test_ascii_timeline_lanes():
    text = ascii_timeline([(0, "rtx", _spans())], step=1, width=20)
    assert "stage 0 compute" in text and "F" in text and "B" in text and ">" in text


def test_breakdown_percentages():
    summary = {"mean_step_s": 1.0, "stages": {0: {"worker": "rtx", "backend": "cuda", "compute_s": 0.6,
                                                   "comm_s": 0.3, "exposed_communication_s": 0.1, "idle_s": 0.2,
                                                   "overlap_ratio": 2 / 3}}}
    b = worker_breakdown(summary)[0]
    assert (b["compute"], b["communication"], b["exposed_communication"], b["idle"]) == (0.6, 0.3, 0.1, 0.2)
    assert "exposed communication" in format_breakdown(summary)

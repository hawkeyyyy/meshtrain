import pytest

from meshtrain.planner.cost import NetworkModel, boundary_transfer_time
from meshtrain.planner.graph import LayerProfile
from meshtrain.planner.partition import PlannerOptions, WorkerProfile, make_plan
from meshtrain.planner.report import format_accuracy, format_placement, prediction_accuracy

GB = 1024**3


def L(i, flops, act, params=1000):
    return LayerProfile(i, f"l{i}", params // 4, params, act, act, act, flops, (1,))


def test_boundary_cost_includes_d2h_and_h2d():
    a = WorkerProfile("a", "a", "cuda", 8 * GB, 1e12, d2h_Bps=1e9, h2d_Bps=1e9)
    m = WorkerProfile("m", "m", "mps", 8 * GB, 1e12, d2h_Bps=2e9, h2d_Bps=5e8)
    c = WorkerProfile("c", "c", "cpu", 8 * GB, 1e11)
    net = NetworkModel(latency={("a", "m"): 0.001, ("a", "c"): 0.001}, bandwidth={("a", "m"): 1e8, ("a", "c"): 1e8})
    t = boundary_transfer_time(1e8, a, m, net)
    assert t["d2h_s"] == pytest.approx(0.1) and t["h2d_s"] == pytest.approx(0.2)
    assert t["network_s"] == pytest.approx(1.001) and t["total_s"] == pytest.approx(1.301)
    assert boundary_transfer_time(1e8, a, c, net)["h2d_s"] == 0.0  # CPU receiver: no copy


def test_async_overlap_changes_stage_time():
    layers = [L(i, 1e9, 10_000_000) for i in range(4)]
    ws = [WorkerProfile("a", "a", "cpu", 8 * GB, 1e10), WorkerProfile("b", "b", "cpu", 8 * GB, 1e10)]
    net = NetworkModel(bandwidth={("a", "b"): 1e7, ("b", "a"): 1e7})
    blocking = make_plan("equal", layers, ws, net, PlannerOptions(async_transport=False))
    overlap = make_plan("equal", layers, ws, net, PlannerOptions(async_transport=True))
    s0b, s0o = blocking.stages[0], overlap.stages[0]
    assert s0b.time_per_mb == pytest.approx(s0b.compute_s + s0b.comm_s)
    assert s0o.time_per_mb == pytest.approx(max(s0o.compute_s, s0o.comm_s))
    assert overlap.predicted_step_s < blocking.predicted_step_s


def test_near_tie_prefers_smaller_boundary():
    # cutting after layer 1 or after layer 2 balances compute equally (layer 2 is ~free),
    # but layer 2's output is 1000x smaller -> topology-aware planner cuts there.
    layers = [L(0, 1e9, 4_000_000), L(1, 1e9, 4_000_000), L(2, 1e6, 4_000), L(3, 1e9, 4_000_000),
              L(4, 1e9, 4_000_000)]
    ws = [WorkerProfile("a", "a", "cpu", 8 * GB, 1e10), WorkerProfile("b", "b", "cpu", 8 * GB, 1e10)]
    net = NetworkModel(bandwidth={("a", "b"): 1e12, ("b", "a"): 1e12}, latency={("a", "b"): 0, ("b", "a"): 0})
    plan = make_plan("topology_aware", layers, ws, net, PlannerOptions(num_stages=2, async_transport=True))
    assert plan.stages[0].end == 3, [(s.start, s.end) for s in plan.stages]
    assert plan.communication_bytes_per_step == 2 * 1 * 4_000


def test_placement_report_and_accuracy():
    layers = [L(i, 1e9, 1_000_000) for i in range(4)]
    ws = [WorkerProfile("rtx", "RTX4070", "cuda", 8 * GB, 1e12, d2h_Bps=6e9, h2d_Bps=6e9),
          WorkerProfile("gtx", "GTX1050Ti", "cuda", 4 * GB, 2e11, d2h_Bps=3e9, h2d_Bps=3e9)]
    plan = make_plan("topology_aware", layers, ws, NetworkModel(), PlannerOptions(num_stages=2, async_transport=True))
    d = plan.to_dict()
    text = format_placement(d)
    for part in ("Stage 0", "worker", "optimizer", "Boundary 0 -> 1", "D2H", "H2D", "Predicted bottleneck"):
        assert part in text
    summary = {"mean_step_s": d["predicted_step_s"] * 1.1,
               "stages": {s["stage"]: {"compute_s": s["compute_s_per_mb"] * d["num_microbatches"] * 2}
                          for s in d["stages"]}}
    acc = prediction_accuracy(d, summary, {"0": {"actual_peak_step0": d["stages"][0]["memory"]["required"] * 1.2}})
    assert acc["step"]["error"] == pytest.approx(1 / 11, rel=1e-6)
    assert acc["stages"][0]["compute_error"] == pytest.approx(0.5)
    assert acc["stages"][0]["memory_error"] == pytest.approx(0.2 / 1.2)
    assert "prediction accuracy" in format_accuracy(acc)

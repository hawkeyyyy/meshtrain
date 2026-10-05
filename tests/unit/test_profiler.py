import threading

from meshtrain.networking.tcp import TCPListener
from meshtrain.profiler.benchmark import benchmark_device, normalise_scores
from meshtrain.profiler.hardware import detect_hardware, primary_backend
from meshtrain.profiler.network import format_matrix, measure_link, serve_probe
from meshtrain.worker.device import CPUDeviceAdapter


def test_detect_hardware_shape():
    hw = detect_hardware()
    assert hw["platform"] in ("linux", "windows", "macos")
    assert hw["cpu_count"] >= 1 and hw["ram_total"] > 0
    assert isinstance(hw["accelerators"], list)
    assert primary_backend(hw) in ("cuda", "mps", "cpu")


def test_cpu_benchmark_measures_positive_throughput():
    b = benchmark_device(CPUDeviceAdapter(), quick=True)
    assert b["matmul_gflops"] > 0 and b["mlp_forward_s"] > 0 and b["mlp_fwd_bwd_s"] > 0
    scores = normalise_scores({"x": b, "y": {**b, "measured_flops": b["measured_flops"] / 2}})
    assert scores == {"x": 1.0, "y": 0.5}


def test_network_probe_over_loopback():
    listener = TCPListener("127.0.0.1", 0)

    def server():
        link, hello = listener.accept(timeout=5)
        assert hello["command_kind"] == "PROBE"
        serve_probe(link)
        link.close()

    th = threading.Thread(target=server)
    th.start()
    m = measure_link("127.0.0.1", listener.port, pings=3, payload_mb=4)
    th.join(5)
    listener.close()
    assert m["bytes"] >= 4_000_000 and m["bandwidth_Bps"] > 0 and m["latency_s"] > 0
    text = format_matrix(["a", "b"], {("a", "b"): 850.0, ("b", "a"): 480.0}, lambda v: f"{v:.0f}Mbps")
    assert "850Mbps" in text and "480Mbps" in text

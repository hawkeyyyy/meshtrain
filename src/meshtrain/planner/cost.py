"""Explicit, documented cost model for pipeline placement (V1.5).

Notation: S stages, M microbatches per step, layer v, worker i.

    network(bytes, i->j)   = latency[i][j] + bytes / bandwidth[i][j]
    transfer(bytes, i->j)  = bytes / d2h[i] + network(bytes, i->j) + bytes / h2d[j]
        (device -> host staging on the sender, the network, host -> device on
         the receiver; d2h/h2d are measured by the worker benchmark and are
         free for CPU workers. CUDA->CUDA and CUDA->MPS therefore differ.)

    compute_time[v][i] = flops[v] * (1 + BACKWARD_FACTOR) / measured_flops[i]
        (backward ~ 2x forward FLOPs for matmul-dominated layers)

    per microbatch, for stage s on worker w_s:
        compute_s = Σ_{v in s} compute_time[v][w_s]
        comm_s    = transfer(act_bytes[out(s)],  w_s -> w_{s+1})   forward activation
                  + transfer(act_bytes[in(s)],   w_s -> w_{s-1})   backward activation-gradient
        stage_time_s = compute_s + comm_s           blocking transport (V1)
                     = max(compute_s, comm_s)       async transport: sends overlap compute (V1.5)

    pipeline_step_time ~= (M + S - 1) * max_s stage_time_s   (fill + drain; same for GPipe and 1F1B)

    communication_per_step = Σ_boundaries 2 * M * act_bytes[boundary]

Placement ties: when two partitions' bottlenecks are within 2%, the one with
fewer boundary bytes wins (less traffic, less exposure to bandwidth noise).

Limitations: link bandwidth is assumed not to be shared between boundaries;
``max(compute, comm)`` assumes perfect overlap (measured overlap ratios in
the timeline show how far reality is from that); ``measured_flops`` comes
from a matmul benchmark, so small/irregular layers are predicted
optimistically. ``prediction_accuracy`` in every run report records
predicted vs actual stage time and memory so these can be calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BACKWARD_FACTOR = 2.0
DEFAULT_LATENCY_S = 1e-3
DEFAULT_BANDWIDTH_BPS = 100e6 / 8  # assume 100 Mbit/s when a link was not measured


@dataclass
class NetworkModel:
    latency: dict[tuple[str, str], float] = field(default_factory=dict)
    bandwidth: dict[tuple[str, str], float] = field(default_factory=dict)  # bytes / second

    def transfer_time(self, nbytes: float, src: str, dst: str) -> float:
        if src == dst:
            return 0.0
        lat = self.latency.get((src, dst), DEFAULT_LATENCY_S)
        bw = self.bandwidth.get((src, dst), DEFAULT_BANDWIDTH_BPS)
        return lat + nbytes / bw

    @classmethod
    def from_measurements(cls, links: dict[str, dict]) -> "NetworkModel":
        """``links`` maps "src->dst" to a measure_link() result."""
        nm = cls()
        for key, m in links.items():
            src, dst = key.split("->")
            nm.latency[(src, dst)] = m["latency_s"]
            nm.bandwidth[(src, dst)] = m["bandwidth_Bps"]
        return nm


def compute_time(flops: float, measured_flops: float) -> float:
    return flops * (1 + BACKWARD_FACTOR) / max(measured_flops, 1.0)


def pipeline_step_time(stage_times_per_mb: list[float], num_microbatches: int) -> float:
    if not stage_times_per_mb:
        return 0.0
    return (num_microbatches + len(stage_times_per_mb) - 1) * max(stage_times_per_mb)


def boundary_transfer_time(nbytes: float, src, dst, network: NetworkModel) -> dict:
    """``src``/``dst`` are WorkerProfiles (need worker_id, d2h_Bps, h2d_Bps)."""
    d2h = nbytes / src.d2h_Bps if src.d2h_Bps else 0.0
    h2d = nbytes / dst.h2d_Bps if dst.h2d_Bps else 0.0
    net = network.transfer_time(nbytes, src.worker_id, dst.worker_id)
    return {"d2h_s": d2h, "network_s": net, "h2d_s": h2d, "total_s": d2h + net + h2d}

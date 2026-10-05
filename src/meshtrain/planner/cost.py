"""Explicit, documented cost model for pipeline placement.

Notation: S stages, M microbatches per step, layer v, worker i.

    transfer_time(bytes, i->j) = latency[i][j] + bytes / bandwidth[i][j]

    compute_time[v][i] = flops[v] * (1 + BACKWARD_FACTOR) / measured_flops[i]
        (backward ~ 2x forward FLOPs for matmul-dominated layers)

    stage_time_per_mb[s] = sum_{v in s} compute_time[v][w_s]
                         + transfer_time(act_bytes[last(s)], w_s -> w_{s+1})   (activation out)
                         + transfer_time(act_bytes[first(s)-1], w_s -> w_{s-1}) (gradient out)

    pipeline_step_time  ~= (M + S - 1) * max_s stage_time_per_mb[s]      (GPipe fill + drain)

    communication_per_step = sum over boundaries of 2 * M * act_bytes[boundary]
                             (activation forward + gradient backward)

The model ignores overlap between compute and communication inside a stage
(V1 sends synchronously) and assumes link bandwidth is not shared between
boundaries. ``measured_flops`` comes from the worker's matmul benchmark, so
small/irregular layers are predicted optimistically.
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

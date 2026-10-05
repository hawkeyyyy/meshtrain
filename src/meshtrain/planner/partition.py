"""Static contiguous-layer partition planner.

Strategies (Experiment 4 compares them):

* ``equal``    -- same number of layers per worker, workers in the given order.
* ``compute``  -- layers split so each worker's share of forward FLOPs is
                  proportional to its measured compute throughput.
* ``auto``     -- memory + compute + network aware: for each candidate ordered
                  subset of eligible workers, a dynamic program over contiguous
                  layer ranges minimises the bottleneck stage time (compute +
                  boundary transfers, see cost.py) subject to every worker's
                  memory budget; ties are broken by total boundary bytes.

All strategies run the same memory check; ``equal``/``compute`` report
violations instead of silently overcommitting a device. A worker never
receives more layers than its memory budget allows in an ``auto`` plan.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from meshtrain.planner.cost import NetworkModel, compute_time, pipeline_step_time
from meshtrain.planner.graph import LayerProfile
from meshtrain.planner.memory import (
    DEFAULT_FRAMEWORK_RESERVE,
    DEFAULT_SAFETY_FACTOR,
    MemoryEstimate,
    estimate_stage_memory,
    reserved_bytes,
)


@dataclass
class WorkerProfile:
    worker_id: str
    name: str
    backend: str
    memory_total: int        # accelerator memory (or RAM for CPU workers)
    measured_flops: float    # from profiler.benchmark
    unified_memory: bool = False
    supported_dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16", "float64")
    compute_score: float = 1.0


@dataclass
class StagePlan:
    stage_index: int
    worker_id: str
    worker_name: str
    backend: str
    start: int
    end: int
    memory: MemoryEstimate
    compute_s: float        # per microbatch, fwd + bwd
    comm_s: float           # per microbatch, outgoing activation + gradient
    boundary_bytes: int     # activation bytes leaving this stage per microbatch

    @property
    def time_per_mb(self) -> float:
        return self.compute_s + self.comm_s


@dataclass
class Plan:
    strategy: str
    stages: list[StagePlan]
    num_microbatches: int
    excluded: dict[str, str] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return bool(self.stages) and not self.violations

    @property
    def bottleneck(self) -> StagePlan | None:
        return max(self.stages, key=lambda s: s.time_per_mb) if self.stages else None

    @property
    def predicted_step_s(self) -> float:
        return pipeline_step_time([s.time_per_mb for s in self.stages], self.num_microbatches)

    @property
    def communication_bytes_per_step(self) -> int:
        return sum(2 * self.num_microbatches * s.boundary_bytes for s in self.stages[:-1])

    @property
    def communication_s_per_step(self) -> float:
        return sum(self.num_microbatches * s.comm_s for s in self.stages)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "feasible": self.feasible,
            "violations": self.violations,
            "excluded": self.excluded,
            "num_microbatches": self.num_microbatches,
            "predicted_step_s": self.predicted_step_s,
            "communication_bytes_per_step": self.communication_bytes_per_step,
            "communication_s_per_step": self.communication_s_per_step,
            "stages": [{
                "stage": s.stage_index, "worker_id": s.worker_id, "worker": s.worker_name, "backend": s.backend,
                "layers": [s.start, s.end], "compute_s_per_mb": s.compute_s, "comm_s_per_mb": s.comm_s,
                "boundary_bytes": s.boundary_bytes, "memory": s.memory.to_dict(),
            } for s in self.stages],
        }

    def format(self, layer_names: list[str] | None = None) -> str:
        gb = lambda n: f"{n / 1024**3:.2f} GB"  # noqa: E731
        lines = [f"Proposed placement ({self.strategy}):", ""]
        for s in self.stages:
            names = ""
            if layer_names:
                names = f" ({layer_names[s.start]} .. {layer_names[s.end - 1]})"
            star = "*" if s.memory.total and s.backend == "mps" else ""
            lines.append(f"  Stage {s.stage_index} -> {s.worker_name} [{s.backend}]")
            lines.append(f"      layers {s.start}-{s.end - 1}{names}")
            lines.append(f"      memory {gb(s.memory.required)} of {gb(s.memory.budget)} usable{star}"
                         f" ({'ok' if s.memory.fits else 'OVER BUDGET'})")
            lines.append(f"      compute {s.compute_s * 1000:.2f} ms/mb, send {s.comm_s * 1000:.2f} ms/mb")
        b = self.bottleneck
        lines += ["", "Predicted:",
                  f"  communication/step: {self.communication_bytes_per_step / 1e6:.2f} MB "
                  f"({self.communication_s_per_step * 1000:.1f} ms)",
                  f"  slowest stage: {b.stage_index} ({b.worker_name}, {b.time_per_mb * 1000:.2f} ms/mb)" if b else "",
                  f"  estimated pipeline time: {self.predicted_step_s * 1000:.1f} ms/step "
                  f"(M={self.num_microbatches}, S={len(self.stages)})"]
        if self.excluded:
            lines.append("  excluded workers: " + ", ".join(f"{w} ({why})" for w, why in self.excluded.items()))
        for v in self.violations:
            lines.append(f"  VIOLATION: {v}")
        return "\n".join(lines)


@dataclass
class PlannerOptions:
    optimizer: str = "sgd"
    num_microbatches: int = 1
    allow_backends: tuple[str, ...] = ("cuda", "mps", "cpu")
    dtype: str = "float32"
    # V1 margin (reserved = max(fraction * total, min)); used only when set.
    headroom_fraction: float | None = None
    headroom_min_bytes: int | None = None
    activation_safety: float = 1.25
    num_stages: int | None = None
    max_orderings: int = 400
    # V1.5 memory model (planner/memory.py)
    schedule: str = "gpipe"                 # conservative default: gpipe holds the most activations
    max_inflight: int | None = None
    safety_factors: dict = field(default_factory=lambda: dict(DEFAULT_SAFETY_FACTOR))
    framework_reserve: dict = field(default_factory=lambda: dict(DEFAULT_FRAMEWORK_RESERVE))
    budget_overrides: dict = field(default_factory=dict)  # worker_id -> total bytes (startup replanning)

    def memory_kwargs(self, backend: str) -> dict:
        if self.headroom_fraction is not None or self.headroom_min_bytes is not None:
            return {"headroom_fraction": self.headroom_fraction, "headroom_min_bytes": self.headroom_min_bytes}
        return {"safety_factor": self.safety_factors.get(backend, 0.85),
                "framework_reserve": self.framework_reserve.get(backend, 0)}


class _Evaluator:
    def __init__(self, layers: list[LayerProfile], network: NetworkModel, opts: PlannerOptions):
        self.layers = layers
        self.network = network
        self.opts = opts
        self.prefix_flops = [0.0]
        for l in layers:
            self.prefix_flops.append(self.prefix_flops[-1] + l.flops)

    def stage(self, idx: int, workers: list[WorkerProfile], start: int, end: int) -> StagePlan:
        w = workers[idx]
        last = idx == len(workers) - 1
        flops = self.prefix_flops[end] - self.prefix_flops[start]
        comp = compute_time(flops, w.measured_flops)
        out_bytes = self.layers[end - 1].activation_bytes if not last else 0
        comm = 0.0
        if not last:
            comm += self.network.transfer_time(out_bytes, w.worker_id, workers[idx + 1].worker_id)
        if idx > 0:
            comm += self.network.transfer_time(self.layers[start].input_bytes, w.worker_id, workers[idx - 1].worker_id)
        total = self.opts.budget_overrides.get(w.worker_id, w.memory_total)
        mem = estimate_stage_memory(self.layers[start:end], total=total, optimizer=self.opts.optimizer,
                                    num_microbatches=self.opts.num_microbatches, is_last=last,
                                    stage_index=idx, num_stages=len(workers), schedule=self.opts.schedule,
                                    max_inflight=self.opts.max_inflight, backend=w.backend,
                                    activation_safety=self.opts.activation_safety,
                                    **self.opts.memory_kwargs(w.backend))
        return StagePlan(idx, w.worker_id, w.name, w.backend, start, end, mem, comp, comm, out_bytes)


def _usable(w: WorkerProfile, opts: PlannerOptions) -> int:
    total = opts.budget_overrides.get(w.worker_id, w.memory_total)
    kw = opts.memory_kwargs(w.backend)
    if "headroom_fraction" in kw:
        return total - reserved_bytes(total, kw["headroom_fraction"] or 0.0, kw["headroom_min_bytes"] or 0)
    return int(total * kw["safety_factor"]) - kw["framework_reserve"]


def eligible_workers(workers: list[WorkerProfile], opts: PlannerOptions) -> tuple[list[WorkerProfile], dict[str, str]]:
    ok, excluded = [], {}
    for w in workers:
        if w.backend not in opts.allow_backends:
            excluded[w.name] = f"backend {w.backend} not allowed"
        elif opts.dtype not in w.supported_dtypes:
            excluded[w.name] = f"no {opts.dtype} support"
        elif _usable(w, opts) <= 0:
            excluded[w.name] = "memory below safety margin"
        else:
            ok.append(w)
    return ok, excluded


def _check(plan: Plan, num_layers: int) -> Plan:
    covered = [s.end - s.start for s in plan.stages]
    if sum(covered) != num_layers or any(c <= 0 for c in covered):
        plan.violations.append("partition does not cover every layer exactly once")
    for s in plan.stages:
        if not s.memory.fits:
            plan.violations.append(
                f"stage {s.stage_index} on {s.worker_name} needs {s.memory.required / 1024**3:.2f} GB "
                f"but only {s.memory.budget / 1024**3:.2f} GB is usable")
    return plan


def _plan_from_bounds(strategy, workers, bounds, ev, excluded) -> Plan:
    stages = [ev.stage(i, workers, a, b) for i, (a, b) in enumerate(bounds)]
    plan = Plan(strategy, stages, ev.opts.num_microbatches, excluded=dict(excluded))
    return _check(plan, len(ev.layers))


def _split_by_weights(n_layers: int, weights: list[float], layer_cost: list[float]) -> list[tuple[int, int]]:
    """Contiguous split of layers so cumulative cost matches cumulative weight share."""
    k = len(weights)
    total_cost = sum(layer_cost) or float(n_layers)
    costs = layer_cost if sum(layer_cost) > 0 else [1.0] * n_layers
    cum_w = list(itertools.accumulate(w / sum(weights) for w in weights))
    bounds, start, acc = [], 0, 0.0
    for i in range(k):
        if i == k - 1:
            end = n_layers
        else:
            target = cum_w[i] * total_cost
            end = start + 1
            acc += costs[start]
            while end < n_layers - (k - 1 - i) and acc + costs[end] / 2 <= target:
                acc += costs[end]
                end += 1
        bounds.append((start, end))
        start = end
    return bounds


def plan_equal(layers, workers, network, opts) -> Plan:
    ws, excluded = eligible_workers(workers, opts)
    k = min(opts.num_stages or len(ws), len(ws), len(layers))
    ws = ws[:k]
    ev = _Evaluator(layers, network, opts)
    n = len(layers)
    bounds = [(n * i // k, n * (i + 1) // k) for i in range(k)]
    return _plan_from_bounds("equal", ws, bounds, ev, excluded)


def plan_compute(layers, workers, network, opts) -> Plan:
    ws, excluded = eligible_workers(workers, opts)
    k = min(opts.num_stages or len(ws), len(ws), len(layers))
    ws = sorted(ws, key=lambda w: -w.measured_flops)[:k]
    ws = [w for w in workers if w in ws]  # keep user order among chosen workers
    ev = _Evaluator(layers, network, opts)
    # cost per layer: flops, with a small floor so parameter-free layers still count
    cost = [max(l.flops, 1.0) for l in layers]
    bounds = _split_by_weights(len(layers), [w.measured_flops for w in ws], cost)
    return _plan_from_bounds("compute", ws, bounds, ev, excluded)


def _dp(ev: _Evaluator, ws: list[WorkerProfile]):
    """min over contiguous splits of (bottleneck time, boundary bytes) with memory feasibility."""
    n, k = len(ev.layers), len(ws)
    INF = (math.inf, math.inf)
    # best[j][s] = (bottleneck, bytes, prev_split) for layers [0, j) on workers [0, s)
    best = [[INF + (None,) for _ in range(k + 1)] for _ in range(n + 1)]
    best[0][0] = (0.0, 0.0, None)
    for s in range(1, k + 1):
        for j in range(s, n - (k - s) + 1):
            cand = INF + (None,)
            for i in range(s - 1, j):
                prev = best[i][s - 1]
                if prev[0] == math.inf:
                    continue
                st = ev.stage(s - 1, ws, i, j)
                if not st.memory.fits:
                    continue
                val = (max(prev[0], st.time_per_mb), prev[1] + st.boundary_bytes, i)
                if val[:2] < cand[:2]:
                    cand = val
            best[j][s] = cand
    if best[n][k][0] == math.inf:
        return None
    bounds, j = [], n
    for s in range(k, 0, -1):
        i = best[j][s][2]
        bounds.append((i, j))
        j = i
    return best[n][k][:2], bounds[::-1]


def plan_auto(layers, workers, network, opts) -> Plan:
    ws, excluded = eligible_workers(workers, opts)
    ev = _Evaluator(layers, network, opts)
    sizes = [opts.num_stages] if opts.num_stages else range(1, min(len(ws), len(layers)) + 1)
    best = None
    tried = 0
    for k in sizes:
        for order in itertools.permutations(ws, k):
            tried += 1
            if tried > opts.max_orderings:
                break
            res = _dp(ev, list(order))
            if res is None:
                continue
            key, bounds = res
            # Same bottleneck: prefer fewer stages (less communication, fewer failure points).
            if best is None or (key[0] < best[0][0] * 0.98) or (abs(key[0] - best[0][0]) <= best[0][0] * 0.02
                                                                and (k, key[1]) < (len(best[1]), best[0][1])):
                best = (key, list(order), bounds)
    if best is None:
        plan = Plan("auto", [], opts.num_microbatches, excluded=excluded)
        plan.violations.append("no contiguous partition fits the memory budgets of the eligible workers")
        return plan
    _, order, bounds = best
    return _plan_from_bounds("auto", order, bounds, ev, excluded)


def plan_manual(layers, workers, network, opts, assignments: list[tuple[str, int, int]]) -> Plan:
    by_id = {w.worker_id: w for w in workers} | {w.name: w for w in workers}
    ws = []
    for wid, _, _ in assignments:
        if wid not in by_id:
            raise ValueError(f"manual placement names unknown worker {wid!r}")
        ws.append(by_id[wid])
    ev = _Evaluator(layers, network, opts)
    return _plan_from_bounds("manual", ws, [(a, b) for _, a, b in assignments], ev, {})


STRATEGIES = {"equal": plan_equal, "compute": plan_compute, "auto": plan_auto}


def make_plan(strategy: str, layers, workers, network: NetworkModel | None = None,
              opts: PlannerOptions | None = None) -> Plan:
    opts = opts or PlannerOptions()
    network = network or NetworkModel()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    return STRATEGIES[strategy](layers, workers, network, opts)

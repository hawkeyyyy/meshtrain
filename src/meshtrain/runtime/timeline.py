"""Execution timeline: intervals of compute, transfer and waiting per worker.

Every interesting phase is recorded as ``[start, end)`` with its category,
thread, step and microbatch. Per-step metrics are *derived from intervals*
rather than accumulated ad hoc, so overlap is measured, not assumed:

    compute_ms               = |union(compute intervals)|
    communication_ms         = |union(transfer intervals, any thread)|
    overlapped_ms            = |compute ∩ communication|
    exposed_communication_ms = communication_ms - overlapped_ms
    idle_ms                  = step_wall - |compute ∪ communication|
    overlap_ratio            = overlapped_ms / communication_ms

Timestamps are wall-clock seconds (``time.time`` anchor + ``perf_counter``
offsets) so traces from several processes on one host line up; across
machines they are only as aligned as the machines' clocks (NTP).

``to_chrome_trace`` writes Chrome Trace Event JSON (open in
chrome://tracing or https://ui.perfetto.dev).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

COMPUTE = ("FORWARD_COMPUTE", "BACKWARD_COMPUTE", "OPTIMIZER_STEP")
TRANSFER = ("D2H_COPY", "SERIALIZE", "QUEUE_WAIT", "NETWORK_SEND", "NETWORK_RECV", "DESERIALIZE", "H2D_COPY")
WAIT = ("WAIT_FORWARD", "WAIT_BACKWARD", "WAIT_SEND")
# QUEUE_WAIT is time a message sat in the outbound queue: it is part of a
# transfer's latency but no work happens, so it is excluded from comm time.
COMM_WORK = tuple(c for c in TRANSFER if c != "QUEUE_WAIT")
# V2 tensor residency (RAM <-> accelerator, runtime/offload.py). Kept apart from
# network communication. Blocking categories happen on the compute thread *inside*
# forward/backward/optimizer spans and are subtracted from compute time.
MEMORY_ASYNC = ("TENSOR_PREFETCH", "REMOTE_PREFETCH", "REMOTE_PUT")
MEMORY_BLOCKING = ("TENSOR_LOAD", "PREFETCH_STALL", "TENSOR_WRITEBACK", "TENSOR_EVICT", "OPTIMIZER_OFFLOAD",
                   "REMOTE_GET", "REMOTE_FETCH_STALL")
MEMORY_STATE = ("TENSOR_ACCELERATOR_RESIDENT", "TENSOR_RAM_RESIDENT", "REMOTE_RESIDENCY")


@dataclass
class Span:
    category: str
    start: float
    end: float
    thread: str
    step: int | None = None
    microbatch: int | None = None
    args: dict | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Timeline:
    def __init__(self, worker: str = "local", stage: int = 0, enabled: bool = True):
        self.worker, self.stage, self.enabled = worker, stage, enabled
        self._anchor_wall = time.time()
        self._anchor_perf = time.perf_counter()
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._anchor_wall + (time.perf_counter() - self._anchor_perf)

    def add(self, category: str, start: float, end: float, step=None, microbatch=None, thread: str | None = None,
            **args) -> None:
        if not self.enabled:
            return
        span = Span(category, start, end, thread or threading.current_thread().name, step, microbatch, args or None)
        with self._lock:
            self._spans.append(span)

    def span(self, category: str, step=None, microbatch=None, **args):
        tl = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = tl.now()
                return self_inner

            def __exit__(self_inner, *exc):
                tl.add(category, self_inner.t0, tl.now(), step, microbatch, **args)

        return _Ctx()

    def spans(self, step: int | None = None) -> list[Span]:
        with self._lock:
            return [s for s in self._spans if step is None or s.step == step]

    def drop_before(self, t: float) -> None:
        """Forget old spans (bounded memory when tracing is off for long runs)."""
        with self._lock:
            self._spans = [s for s in self._spans if s.end >= t]

    # -- analysis ---------------------------------------------------------
    def step_metrics(self, step: int, window: tuple[float, float]) -> dict:
        lo, hi = window
        spans = [s for s in self.spans() if s.end > lo and s.start < hi]

        def clip(cats):
            return [(max(s.start, lo), min(s.end, hi)) for s in spans if s.category in cats]

        mem_block = _union(clip(MEMORY_BLOCKING))
        compute = _subtract(_union(clip(COMPUTE)), mem_block)
        comm = _union(clip(COMM_WORK))
        mem_async = _union(clip(MEMORY_ASYNC))
        busy = _union(compute + comm + mem_block)
        compute_s, comm_s = _length(compute), _length(comm)
        overlapped = _length(_intersect(compute, comm))
        wall = hi - lo
        per_cat = {}
        for s in spans:
            if s.category in MEMORY_STATE:
                continue
            if s.step == step or s.category in COMPUTE + COMM_WORK + MEMORY_ASYNC + MEMORY_BLOCKING:
                per_cat[s.category] = per_cat.get(s.category, 0.0) + (min(s.end, hi) - max(s.start, lo))
        mem_overlapped = _length(_intersect(mem_async, compute))
        mem_total = _length(_union(mem_async + mem_block))
        return {
            "memory_transfer_s": mem_total,
            "memory_overlapped_s": mem_overlapped,
            "exposed_memory_transfer_s": _length(mem_block) + max(0.0, _length(mem_async) - mem_overlapped
                                                                   - _length(_intersect(mem_async, mem_block))),
            "compute_s": compute_s,
            "communication_s": comm_s,
            "overlapped_s": overlapped,
            "exposed_communication_s": comm_s - overlapped,
            "idle_s": max(0.0, wall - _length(busy)),
            "overlap_ratio": overlapped / comm_s if comm_s > 0 else 0.0,
            "phase_s": per_cat,
        }

    # -- export -----------------------------------------------------------
    def to_chrome_events(self, pid: int | None = None) -> list[dict]:
        pid = self.stage if pid is None else pid
        tids: dict[str, int] = {}
        events = [{"ph": "M", "name": "process_name", "pid": pid, "args": {"name": f"stage {self.stage} · {self.worker}"}}]
        for s in self.spans():
            tid = tids.setdefault(s.thread, len(tids))
            name = s.category if s.microbatch is None else f"{s.category} mb{s.microbatch}"
            events.append({"name": name, "cat": s.category, "ph": "X", "pid": pid, "tid": tid,
                           "ts": s.start * 1e6, "dur": s.duration * 1e6,
                           "args": {"step": s.step, "microbatch": s.microbatch, **(s.args or {})}})
        for thread, tid in tids.items():
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": thread}})
        return events

    def export(self) -> list[dict]:
        return [{"category": s.category, "start": s.start, "end": s.end, "thread": s.thread, "step": s.step,
                 "microbatch": s.microbatch, "args": s.args} for s in self.spans()]


def write_chrome_trace(path: str | Path, stage_events: list[list[dict]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [e for evs in stage_events for e in evs]
    path.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}))
    return path


def chrome_events_from_export(spans: list[dict], stage: int, worker: str) -> list[dict]:
    tl = Timeline(worker, stage)
    tl._spans = [Span(s["category"], s["start"], s["end"], s["thread"], s.get("step"), s.get("microbatch"),
                      s.get("args")) for s in spans]
    return tl.to_chrome_events()


# -- interval arithmetic ------------------------------------------------------
def _union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(i for i in iv if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, i, j = [], 0, 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """x minus y (both unions of disjoint sorted intervals)."""
    out = []
    for a, b in x:
        cur = a
        for c, d in y:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def _length(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)

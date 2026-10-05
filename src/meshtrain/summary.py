"""Summaries of runs/<run-id>/metrics.jsonl."""

from __future__ import annotations

import statistics
from collections import defaultdict


def summarize(records: list[dict], warmup_steps: int = 1) -> dict:
    """Aggregate STEP_COMPLETE records (one per stage per step)."""
    steps = [r for r in records if r.get("event") == "STEP_COMPLETE"]
    if not steps:
        return {"steps": 0}
    by_stage: dict[int, list[dict]] = defaultdict(list)
    for r in steps:
        by_stage[r["stage"]].append(r)
    first = sorted(by_stage[min(by_stage)], key=lambda r: r["step"])
    losses = [r["loss"] for r in first if "loss" in r]
    measured = first[warmup_steps:] or first
    step_s = statistics.mean(r["step_s"] for r in measured)
    stages = {}
    for s, rs in sorted(by_stage.items()):
        rs_m = sorted(rs, key=lambda r: r["step"])[warmup_steps:] or rs
        mean = lambda k: statistics.mean(r.get(k, 0.0) for r in rs_m)  # noqa: E731
        mem = rs_m[-1].get("memory", {})
        stages[s] = {
            "worker": rs_m[-1].get("worker"),
            "backend": rs_m[-1].get("backend"),
            "forward_s": mean("forward_s"),
            "backward_s": mean("backward_s"),
            "optimizer_s": mean("optimizer_s"),
            "comm_s": mean("comm_s"),
            "idle_s": mean("idle_s"),
            "utilization": mean("utilization"),
            "bytes_sent_per_step": mean("bytes_sent"),
            "memory": mem,
            "lifecycle_s": {k: statistics.mean(r.get("lifecycle", {}).get(k, 0.0) for r in rs_m)
                            for k in sorted({k for r in rs_m for k in r.get("lifecycle", {})})},
        }
    return {
        "steps": len(first),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "mean_step_s": step_s,
        "steps_per_s": 1.0 / step_s if step_s else 0.0,
        "samples_per_s": statistics.mean(r.get("samples_per_s", 0.0) for r in measured),
        "bytes_per_step": sum(st["bytes_sent_per_step"] for st in stages.values()),
        "stages": stages,
    }


def format_summary(s: dict) -> str:
    if not s.get("steps"):
        return "no steps recorded"
    lines = [
        f"steps: {s['steps']}   initial loss: {s['initial_loss']:.4f}   final loss: {s['final_loss']:.4f}",
        f"step time: {s['mean_step_s'] * 1000:.1f} ms   ({s['steps_per_s']:.2f} steps/s, "
        f"{s['samples_per_s']:.1f} samples/s)   network: {s['bytes_per_step'] / 1e6:.2f} MB/step",
        "",
        f"{'stage':<6}{'worker':<16}{'backend':<8}{'fwd ms':>9}{'bwd ms':>9}{'comm ms':>9}{'idle ms':>9}"
        f"{'util':>7}{'params MB':>11}{'act peak MB':>12}",
    ]
    for idx, st in s["stages"].items():
        mem = st.get("memory", {})
        lines.append(
            f"{idx:<6}{str(st['worker'])[:15]:<16}{str(st['backend']):<8}{st['forward_s'] * 1000:>9.1f}"
            f"{st['backward_s'] * 1000:>9.1f}{st['comm_s'] * 1000:>9.1f}{st['idle_s'] * 1000:>9.1f}"
            f"{st['utilization'] * 100:>6.0f}%{mem.get('parameters', 0) / 1e6:>11.2f}"
            f"{mem.get('saved_activations_peak', 0) / 1e6:>12.2f}")
    return "\n".join(lines)

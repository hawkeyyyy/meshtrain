"""V2 benchmarks: offload strategies, memory budgets, prefetch distance and
capacity (largest trainable model) on one device.

Every trial runs in a fresh process (``spawn``) so allocator state and OOMs
never leak between trials. On CUDA the caching allocator is hard-capped at
the budget (``torch.cuda.set_per_process_memory_fraction``), so a static
model that does not fit really fails with CUDA OOM, and the Windows driver
cannot silently spill into shared system memory.

Strategies (docs/v2-architecture.md):

    static                 A  V1.5: everything resident, AdamW on the device
    local-offload          B  layers in RAM between uses, AdamW on the device, synchronous loads
    optimizer-offload      C  B + AdamW state/master weights in RAM, CPU optimizer step
    local-offload-prefetch D  C + prefetch distance 1 (transfers overlap compute)
    auto-offload              planner keeps as many layers resident as the budget allows, C/D otherwise

Outputs (machine-readable, no plotting in the runtime): ``runs/<name>/results.json``
and ``results.csv`` with one row per trial.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

GB, MB = 1024**3, 1024**2

STRATEGIES = {
    "static": dict(strategy="static"),
    "local-offload": dict(strategy="manual_offload", prefetch_distance=0),
    "optimizer-offload": dict(strategy="manual_offload", prefetch_distance=0, optimizer_execution="cpu_offload"),
    "local-offload-prefetch": dict(strategy="manual_offload", prefetch_distance=1, optimizer_execution="cpu_offload"),
    "auto-offload": dict(strategy="auto_offload", prefetch_distance=1, optimizer_execution="cpu_offload"),
}
LETTER = {"static": "A", "local-offload": "B", "optimizer-offload": "C", "local-offload-prefetch": "D",
          "auto-offload": "auto"}


def capacity_model(blocks: int, hidden: int = 1280) -> dict:
    """Same family as the V1.5 baseline (scripts/v15_baseline.py capacity)."""
    return {"type": "tiny_transformer", "layers": blocks, "hidden_size": hidden, "heads": 8, "vocab_size": 1024,
            "seq_len": 64}


def _trial(model_cfg: dict, strategy: str, device: str, budget: int | None, steps: int, batch: int, micro: int,
           overrides: dict | None) -> dict:
    os.environ["MESHTRAIN_QUIET"] = "1"
    import psutil
    import torch

    from meshtrain.experiments.offload_correctness import build_single_stage
    from meshtrain.runtime.offload import ResidencyPolicy
    from meshtrain.runtime.pipeline import PipelineSettings, run_stage

    kw = {**STRATEGIES[strategy], **(overrides or {})}
    policy = ResidencyPolicy(**kw)
    out = {"strategy": strategy, "policy": kw, "model": model_cfg, "device": device, "budget": budget,
           "batch": batch, "microbatches": batch // micro, "steps": steps}
    proc = psutil.Process()
    try:
        from meshtrain.models import build_model_spec

        out["parameters"] = build_model_spec(model_cfg).parameter_count()
        t_build = time.perf_counter()
        stage, spec = build_single_stage(model_cfg, device=device, policy=policy, budget_bytes=budget,
                                         optimizer="adamw", lr=3e-4, microbatch_size=micro)
        out["build_s"] = time.perf_counter() - t_build
        if stage.residency is not None:
            out["hot_layers"] = len(stage.residency.hot)
            out["cold_layers"] = len(stage.residency.cold)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        recs, losses = [], []
        rss_peak = proc.memory_info().rss
        for i in range(steps):
            s = PipelineSettings("v2-bench", steps=1, batch_size=batch, num_microbatches=batch // micro,
                                 step_offset=i, log_every=10**6, schedule="1f1b", trace=False)
            res = run_stage(stage, spec, s, upstream=None, downstream=None, worker="bench")
            recs.append(res.step_metrics[0])
            losses.append(res.losses[0])
            rss_peak = max(rss_peak, proc.memory_info().rss)
        steady = recs[1:] or recs
        mean = lambda k: sum(r.get(k, 0.0) for r in steady) / len(steady)  # noqa: E731
        step_s = mean("step_s")
        res_stats = [r.get("residency") or {} for r in steady]
        rmean = lambda k: sum(x.get(k, 0) for x in res_stats) / len(res_stats)  # noqa: E731
        out.update(
            ok=True, loss_first=losses[0], loss_last=losses[-1], losses=losses, step_s=step_s,
            steps_per_s=1.0 / step_s if step_s else 0.0, samples_per_s=batch / step_s if step_s else 0.0,
            compute_s=mean("compute_s"), optimizer_s=mean("optimizer_s"),
            memory_transfer_s=mean("memory_transfer_s"), exposed_memory_transfer_s=mean("exposed_memory_transfer_s"),
            memory_overlapped_s=mean("memory_overlapped_s"),
            accelerator_peak_allocated=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
            accelerator_peak_reserved=int(torch.cuda.max_memory_reserved()) if device == "cuda" else None,
            ledger_accelerator_peak=max(r["tensor_store"]["accelerator_peak"] for r in recs),
            ram_resident=recs[-1]["tensor_store"]["ram_resident"],
            pinned_bytes=recs[-1]["tensor_store"]["pinned_bytes"],
            process_rss_peak=rss_peak,
            H2D_bytes_per_step=rmean("H2D_bytes"), D2H_bytes_per_step=rmean("D2H_bytes"),
            H2D_ms_per_step=rmean("H2D_ms"), D2H_ms_per_step=rmean("D2H_ms"),
            prefetch_stall_ms=rmean("prefetch_stall_ms"), sync_load_ms=rmean("sync_load_ms"),
            tensor_cache_hit_ratio=rmean("tensor_cache_hit_ratio") if res_stats[0] else None,
            evictions_per_step=rmean("tensor_eviction_count"), budget_violations=rmean("budget_violations"),
        )
        if budget is not None and device == "cuda":
            out["within_budget"] = out["accelerator_peak_reserved"] <= budget
        stage.close()
    except Exception as exc:
        msg = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
        out.update(ok=False, error=f"{type(exc).__name__}: {msg}",
                   oom="out of memory" in msg.lower() or type(exc).__name__ in ("OutOfMemoryError", "MemoryBudgetError",
                                                                             "MemoryError"))
    return out


def _child(q, *a):
    try:
        q.put(_trial(*a))
    except BaseException as exc:  # pragma: no cover - last resort
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def run_trial(model_cfg: dict, strategy: str, *, device: str = "cuda", budget: int | None = None, steps: int = 3,
              batch: int = 8, micro: int = 2, overrides: dict | None = None, timeout: float = 1800.0) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, model_cfg, strategy, device, budget, steps, batch, micro, overrides))
    t0 = time.time()
    p.start()
    try:
        r = q.get(timeout=timeout)
    except Exception:
        r = {"ok": False, "error": f"trial timed out or crashed (exit code {p.exitcode})", "strategy": strategy,
             "model": model_cfg, "budget": budget, "device": device}
    finally:
        p.join(30)
        if p.is_alive():
            p.terminate()
    r["wall_s"] = time.time() - t0
    return r


CSV_FIELDS = ["strategy", "letter", "parameters", "blocks", "hidden", "budget_mb", "prefetch_distance", "ok",
              "step_s", "steps_per_s", "samples_per_s", "accelerator_peak_allocated_mb", "accelerator_peak_reserved_mb",
              "ledger_accelerator_peak_mb", "ram_resident_mb", "process_rss_peak_mb", "H2D_mb_per_step",
              "D2H_mb_per_step", "exposed_memory_transfer_s", "memory_overlapped_s", "prefetch_stall_ms",
              "tensor_cache_hit_ratio", "hot_layers", "cold_layers", "loss_first", "loss_last", "error"]


def csv_row(r: dict) -> dict:
    m = r.get("model") or {}
    f = lambda k: (r.get(k) / MB) if r.get(k) is not None else None  # noqa: E731
    return {"strategy": r.get("strategy"), "letter": LETTER.get(r.get("strategy"), ""),
            "parameters": r.get("parameters"), "blocks": m.get("layers"), "hidden": m.get("hidden_size"),
            "budget_mb": (r["budget"] / MB) if r.get("budget") else None,
            "prefetch_distance": (r.get("policy") or {}).get("prefetch_distance"), "ok": r.get("ok"),
            "step_s": r.get("step_s"), "steps_per_s": r.get("steps_per_s"), "samples_per_s": r.get("samples_per_s"),
            "accelerator_peak_allocated_mb": f("accelerator_peak_allocated"),
            "accelerator_peak_reserved_mb": f("accelerator_peak_reserved"),
            "ledger_accelerator_peak_mb": f("ledger_accelerator_peak"), "ram_resident_mb": f("ram_resident"),
            "process_rss_peak_mb": f("process_rss_peak"), "H2D_mb_per_step": f("H2D_bytes_per_step"),
            "D2H_mb_per_step": f("D2H_bytes_per_step"), "exposed_memory_transfer_s": r.get("exposed_memory_transfer_s"),
            "memory_overlapped_s": r.get("memory_overlapped_s"), "prefetch_stall_ms": r.get("prefetch_stall_ms"),
            "tensor_cache_hit_ratio": r.get("tensor_cache_hit_ratio"), "hot_layers": r.get("hot_layers"),
            "cold_layers": r.get("cold_layers"), "loss_first": r.get("loss_first"), "loss_last": r.get("loss_last"),
            "error": r.get("error")}


def write_results(name: str, rows: list[dict], meta: dict, runs_dir: str = "runs") -> Path:
    d = Path(runs_dir) / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"meta": meta, "trials": rows}, indent=2, default=str),
                                    encoding="utf-8")
    with open(d / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(csv_row(r))
    return d


def environment(device: str) -> dict:
    import platform

    import psutil
    import torch

    env = {"host": platform.node(), "os": f"{platform.system()} {platform.release()}", "torch": torch.__version__,
           "ram_total": psutil.virtual_memory().total, "date": datetime.now().isoformat(timespec="seconds")}
    if device == "cuda" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_total"] = torch.cuda.get_device_properties(0).total_memory
    return env


def _fmt(r: dict) -> str:
    if not r.get("ok"):
        return f"FAILED: {r.get('error', '')[:110]}"
    peak = r.get("accelerator_peak_reserved") or r.get("ledger_accelerator_peak") or 0
    return (f"{r['step_s'] * 1000:8.0f} ms/step  {r['samples_per_s']:7.1f} samples/s  peak {peak / GB:5.2f} GB  "
            f"RAM {r.get('ram_resident', 0) / GB:5.2f} GB  H2D {r.get('H2D_bytes_per_step', 0) / GB:6.2f} GB/step  "
            f"exposed {r.get('exposed_memory_transfer_s', 0) * 1000:6.0f} ms")


# -- benchmark: strategies A-D at one model + budget -----------------------------------------
def run_offload_benchmark(model_cfg: dict, *, device: str = "cuda", budget: int | None = None, steps: int = 4,
                          batch: int = 8, micro: int = 2, strategies: list[str] | None = None,
                          echo=print) -> tuple[list[dict], Path]:
    rows = []
    for s in strategies or list(STRATEGIES):
        r = run_trial(model_cfg, s, device=device, budget=budget, steps=steps, batch=batch, micro=micro)
        echo(f"  {LETTER[s]:>4} {s:<24} {_fmt(r)}")
        rows.append(r)
    d = write_results("offload-bench", rows, {"kind": "strategies", "model": model_cfg, "budget": budget,
                                              "environment": environment(device)})
    return rows, d


def run_budget_sweep(model_cfg: dict, budgets: list[int], *, device: str = "cuda", strategy: str = "auto-offload",
                     steps: int = 4, batch: int = 8, micro: int = 2, echo=print) -> tuple[list[dict], Path]:
    rows = []
    for b in budgets:
        r = run_trial(model_cfg, strategy, device=device, budget=b, steps=steps, batch=batch, micro=micro)
        echo(f"  budget {b / GB:5.2f} GB  {_fmt(r)}")
        rows.append(r)
    d = write_results("offload-budget-sweep", rows, {"kind": "budget_sweep", "model": model_cfg,
                                                     "strategy": strategy, "environment": environment(device)})
    return rows, d


def run_prefetch_sweep(model_cfg: dict, distances: list[int], *, device: str = "cuda", budget: int | None = None,
                       strategy: str = "local-offload-prefetch", steps: int = 4, batch: int = 8, micro: int = 2,
                       echo=print) -> tuple[list[dict], Path]:
    rows = []
    for dist in distances:
        r = run_trial(model_cfg, strategy, device=device, budget=budget, steps=steps, batch=batch, micro=micro,
                      overrides={"prefetch_distance": dist})
        echo(f"  prefetch {dist}  {_fmt(r)}")
        rows.append(r)
    d = write_results("offload-prefetch-sweep", rows, {"kind": "prefetch_sweep", "model": model_cfg,
                                                       "budget": budget, "strategy": strategy,
                                                       "environment": environment(device)})
    return rows, d


# -- capacity: largest trainable model per strategy ----------------------------------------
def run_capacity(strategies: list[str], *, device: str = "cuda", budget: int | None = None,
                 blocks: list[int] | None = None, hidden: int = 1280, steps: int = 3, batch: int = 8, micro: int = 2,
                 echo=print) -> tuple[dict, Path]:
    blocks = blocks or [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
    rows, best = [], {}
    for s in strategies:
        for b in blocks:
            r = run_trial(capacity_model(b, hidden), s, device=device, budget=budget, steps=steps, batch=batch,
                          micro=micro)
            r["blocks"] = b
            rows.append(r)
            echo(f"  {s:<24} {b:>3} blocks {r.get('parameters', 0) / 1e6:7.1f}M  {_fmt(r)}")
            if not r.get("ok"):
                break
            best[s] = r
    summary = capacity_summary(strategies, best, rows)
    d = write_results("offload-capacity", rows, {"kind": "capacity", "budget": budget, "hidden": hidden,
                                                 "blocks": blocks, "summary": summary,
                                                 "environment": environment(device)})
    return summary, d


def capacity_summary(strategies: list[str], best: dict, rows: list[dict]) -> dict:
    out = {}
    base = best.get("static")
    for s in strategies:
        b = best.get(s)
        fail = next((r for r in rows if r["strategy"] == s and not r.get("ok")), None)
        entry = {"P_max": b["parameters"] if b else 0, "steps_per_s_at_P_max": b["steps_per_s"] if b else None,
                 "blocks_at_P_max": b.get("blocks") if b else None,
                 "first_failure": {"parameters": fail.get("parameters"), "error": fail.get("error")} if fail else None}
        if base and b and s != "static":
            entry["capacity_gain"] = b["parameters"] / base["parameters"]
            # throughput cost: static throughput at its own P_max vs this strategy at its P_max (models differ),
            # plus the like-for-like slowdown at the static P_max size.
            same = next((r for r in rows if r["strategy"] == s and r.get("ok")
                         and r.get("blocks") == base.get("blocks")), None)
            if same:
                entry["slowdown_at_static_P_max"] = base["steps_per_s"] / same["steps_per_s"]
        out[s] = entry
    return out


def format_capacity(summary: dict, budget: int | None) -> str:
    lines = [f"Capacity (largest trainable model), budget {budget / GB:.2f} GB" if budget else "Capacity"]
    for s, e in summary.items():
        line = f"  {s:<24} P_max {e['P_max'] / 1e6:8.1f}M"
        if e.get("steps_per_s_at_P_max"):
            line += f"  {e['steps_per_s_at_P_max']:.3f} steps/s"
        if e.get("capacity_gain"):
            line += f"  capacity gain {e['capacity_gain']:.2f}x"
        if e.get("slowdown_at_static_P_max"):
            line += f"  slowdown at static P_max {e['slowdown_at_static_P_max']:.2f}x"
        if e.get("first_failure"):
            ff = e["first_failure"]
            line += f"  (fails at {(ff.get('parameters') or 0) / 1e6:.1f}M: {str(ff.get('error'))[:60]})"
        lines.append(line)
    return "\n".join(lines)

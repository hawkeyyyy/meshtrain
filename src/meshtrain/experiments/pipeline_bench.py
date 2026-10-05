"""``meshtrain benchmark pipeline``: V1 vs V1.5 execution modes on the same model.

Modes (same model, data, placement and seeds):

    V1     gpipe  + blocking sends
           gpipe  + async sends
           1f1b   + blocking sends
    V1.5   1f1b   + async sends   (+ buffer reuse, pinned staging on CUDA)

Locally every stage is its own process. Loopback TCP is far faster than a
real LAN, so by default the links are *emulated* (bandwidth/latency
throttle, see networking/emulation.py) -- every report says which link was
used. With ``--cluster`` the modes are submitted as jobs to the running
coordinator instead (real devices, real network).

Reported per mode: steps/s, samples/s, step time, peak saved activation
bytes (max over stages and stage 0), idle, communication, exposed
communication, overlap ratio, bytes sent per step, loss, and whether the
loss trajectory matches the V1 mode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from meshtrain.experiments.results_doc import environment_line, md_table, update_section
from meshtrain.models import build_model_spec
from meshtrain.runtime.local import LocalStage, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings
from meshtrain.runtime.trace_report import ascii_timeline, format_breakdown, write_run_timeline
from meshtrain.summary import summarize
from meshtrain.telemetry import new_run_id

MODES = [
    ("V1: gpipe + blocking", "gpipe", False),
    ("gpipe + async", "gpipe", True),
    ("1f1b + blocking", "1f1b", False),
    ("V1.5: 1f1b + async", "1f1b", True),
]
DEFAULT_MODEL = {"type": "tiny_transformer", "layers": 6, "hidden_size": 256, "heads": 4, "vocab_size": 512,
                 "seq_len": 64}
DOC = Path(__file__).resolve().parents[3] / "docs" / "v1-vs-v1.5.md"


def _row(name: str, s: dict, batch: int, losses: list[float], ref_losses: list[float] | None) -> dict:
    st = s["stages"]
    step = s["mean_step_s"]
    agg = lambda k: sum(v.get(k, 0.0) for v in st.values())  # noqa: E731
    comm = agg("comm_s")
    return {
        "mode": name, "steps_per_s": 1 / step, "samples_per_s": batch / step, "step_ms": step * 1000,
        "peak_act_MB_max": max(v["peak_saved_activation_bytes"] for v in st.values()) / 1e6,
        "peak_act_MB_stage0": st[min(st)]["peak_saved_activation_bytes"] / 1e6,
        "peak_saved_mb_stage0": st[min(st)]["peak_saved_microbatches"],
        "idle_ms": agg("idle_s") / len(st) * 1000, "comm_ms": comm / len(st) * 1000,
        "exposed_ms": agg("exposed_communication_s") / len(st) * 1000,
        "overlap_ratio": agg("overlapped_s") / comm if comm else 0.0,
        "MB_per_step": s["bytes_per_step"] / 1e6, "final_loss": losses[-1],
        "loss_matches_v1": None if ref_losses is None else
        max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(losses, ref_losses)) < 1e-5,
    }


def run_local_benchmark(model_cfg: dict | None = None, num_stages: int = 3, batch_size: int = 32,
                        num_microbatches: int = 8, steps: int = 8, bandwidth_mbps: float | None = 100.0,
                        latency_ms: float = 1.0, runs_dir: str = "runs", write_doc: bool = True,
                        devices: list[str] | None = None) -> dict:
    model_cfg = model_cfg or DEFAULT_MODEL
    n = build_model_spec(model_cfg).num_layers
    devices = devices or ["cpu"] * num_stages
    stages = [LocalStage((n * i // num_stages, n * (i + 1) // num_stages), devices[i]) for i in range(num_stages)]
    link = None if not bandwidth_mbps else (bandwidth_mbps * 1e6 / 8, latency_ms / 1000)
    run_id = new_run_id("pipeline-bench")
    run_dir = Path(runs_dir) / run_id
    rows, ref, v15 = [], None, None
    for name, schedule, asyn in MODES:
        settings = PipelineSettings(f"bench-{schedule}-{int(asyn)}", steps=steps, batch_size=batch_size,
                                    num_microbatches=num_microbatches, schedule=schedule, async_transport=asyn,
                                    log_every=10_000)
        res = run_local_pipeline(model_cfg, stages, settings, optimizer="adamw", lr=3e-4, transport="tcp",
                                 link_emulation=link)
        s = summarize([m for r in res for m in r.step_metrics], warmup_steps=2)
        row = _row(name, s, batch_size, res[0].losses, ref)
        rows.append(row)
        ref = ref or res[0].losses
        lanes = [(r.stage_index, r.worker, r.timeline or []) for r in res]
        tag = f"{schedule}-{'async' if asyn else 'blocking'}"
        write_run_timeline(run_dir / f"timeline-{tag}.json", lanes)
        if name.startswith("V1.5"):
            v15 = {"summary": s, "lanes": lanes, "settings": settings}
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = build_model_spec(model_cfg)
    meta = {"model": model_cfg, "parameters": spec.parameter_count(), "stages": [list(s.layers) for s in stages],
            "devices": devices, "batch_size": batch_size, "microbatches": num_microbatches, "steps": steps,
            "link": "loopback TCP (not emulated)" if link is None else
            f"emulated {bandwidth_mbps:g} Mbit/s, {latency_ms:g} ms per message, each direction"}
    (run_dir / "benchmark.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    gantt = ascii_timeline(v15["lanes"], step=steps - 1)
    breakdown = format_breakdown(v15["summary"])
    (run_dir / "timeline_report.txt").write_text(breakdown + "\n\n" + gantt + "\n")
    md = render_markdown(meta, rows, breakdown, gantt, run_id)
    if write_doc:
        update_section("pipeline-benchmark-local", md, DOC)
    return {"run_dir": str(run_dir), "meta": meta, "rows": rows, "markdown": md}


def render_markdown(meta: dict, rows: list[dict], breakdown: str, gantt: str, run_id: str) -> str:
    v1, v15 = rows[0], rows[-1]
    table = md_table(
        ["mode", "steps/s", "samples/s", "step ms", "peak act MB (max stage)", "peak act MB (stage 0)",
         "saved mb (stage 0)", "idle ms", "comm ms", "exposed comm ms", "overlap", "MB/step", "final loss",
         "same loss as V1"],
        [[r["mode"], f"{r['steps_per_s']:.2f}", f"{r['samples_per_s']:.1f}", f"{r['step_ms']:.0f}",
          f"{r['peak_act_MB_max']:.1f}", f"{r['peak_act_MB_stage0']:.1f}", r["peak_saved_mb_stage0"],
          f"{r['idle_ms']:.0f}", f"{r['comm_ms']:.0f}", f"{r['exposed_ms']:.0f}", f"{r['overlap_ratio']:.0%}",
          f"{r['MB_per_step']:.2f}", f"{r['final_loss']:.4f}",
          "-" if r["loss_matches_v1"] is None else ("yes" if r["loss_matches_v1"] else "NO")] for r in rows])
    return "\n\n".join([
        environment_line(),
        f"Run `{run_id}`. Model: {meta['model']['type']} ({meta['parameters'] / 1e6:.2f}M params), stages "
        f"{meta['stages']} on {', '.join(meta['devices'])}, batch {meta['batch_size']} = {meta['microbatches']} "
        f"microbatches, {meta['steps']} steps (first 2 excluded from averages). Link: **{meta['link']}**.",
        table,
        f"**V1 → V1.5:** {v1['step_ms']:.0f} → {v15['step_ms']:.0f} ms/step "
        f"({v1['step_ms'] / v15['step_ms']:.2f}x throughput); exposed communication "
        f"{v1['exposed_ms']:.0f} → {v15['exposed_ms']:.0f} ms/stage/step; stage-0 peak saved activations "
        f"{v1['peak_act_MB_stage0']:.1f} → {v15['peak_act_MB_stage0']:.1f} MB.",
        "Per-worker breakdown (V1.5 mode):\n\n```\n" + breakdown + "\n```",
        "Last step of the V1.5 mode:\n\n```\n" + gantt + "\n```",
    ])


def run_cluster_benchmark(client, config_raw: dict, poll_s: float = 1.0, write_doc: bool = True) -> dict:
    """Submit the same config once per mode to the running cluster (real devices + network)."""
    rows, ref = [], None
    batch = config_raw["training"]["batch_size"]
    for name, schedule, asyn in MODES:
        cfg = json.loads(json.dumps(config_raw))
        cfg["job"]["name"] = f"bench-{schedule}-{'async' if asyn else 'blocking'}"
        cfg.setdefault("pipeline", {})["schedule"] = schedule
        cfg.setdefault("transport", {})["async"] = asyn
        job = client.start_job(cfg)
        while True:
            j = client.job(job["job_id"])
            if j["status"] != "RUNNING":
                break
            time.sleep(poll_s)
        if j["status"] != "COMPLETED" or not j.get("summary"):
            rows.append({"mode": name, "error": j.get("error")})
            continue
        losses = [l for _, l in sorted(j["losses"])]
        s = j["summary"]
        s["stages"] = {int(k): v for k, v in s["stages"].items()}
        rows.append({**_row(name, s, batch, losses, ref), "job": j["job_id"]})
        ref = ref or losses
    ok = [r for r in rows if "error" not in r]
    md = "\n\n".join([environment_line(), "Cluster run (real devices and network) via the coordinator.",
                      md_table(["mode", "job", "steps/s", "step ms", "peak act MB (stage 0)", "exposed comm ms",
                                "overlap", "MB/step", "final loss"],
                               [[r["mode"], r.get("job", "-"), f"{r['steps_per_s']:.2f}", f"{r['step_ms']:.0f}",
                                 f"{r['peak_act_MB_stage0']:.1f}", f"{r['exposed_ms']:.0f}",
                                 f"{r['overlap_ratio']:.0%}", f"{r['MB_per_step']:.2f}", f"{r['final_loss']:.4f}"]
                                for r in ok])]
                     + [f"- {r['mode']}: FAILED {r['error']}" for r in rows if "error" in r])
    if write_doc:
        update_section("pipeline-benchmark-cluster", md, DOC)
    return {"rows": rows, "markdown": md}


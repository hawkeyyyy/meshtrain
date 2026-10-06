"""Experiment 5 — capacity: largest model trainable on each device alone vs. on the mesh.

    capacity_gain     = P_mesh_max / max_i(P_single_max[i])
    throughput_penalty = steps/s(best single device, its largest model)
                         / steps/s(mesh, its largest model)

Modes
-----
``emulated`` (runs anywhere): the target cluster (RTX 8 GB, RTX 12 GB, M2 with
16 GB unified memory) is scaled down by ``SCALE`` and emulated by CPU
processes. Feasibility uses the planner's memory accounting against the
emulated budgets; the largest models are then actually trained and the
tensor bytes each stage held are checked against its budget.

``hardware`` (needs a running coordinator with real workers): every model
size is actually submitted -- first to each worker alone, then to the mesh
with ``placement: auto`` -- and an out-of-memory failure marks the limit.
"""

from __future__ import annotations

import statistics
import time

from meshtrain.experiments.emulation import MB, EmulatedWorker, profile_emulated
from meshtrain.experiments.results_doc import environment_line, md_table, update_section
from meshtrain.models import build_model_spec
from meshtrain.planner.cost import NetworkModel
from meshtrain.planner.graph import profile_model
from meshtrain.planner.partition import PlannerOptions, make_plan
from meshtrain.runtime.local import LocalStage, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings

GB = 1024**3
SCALE = 32  # emulated budgets = real device memory / SCALE
TARGET = [  # (name, real memory in bytes); M2 Air 16 GB: ~2/3 usable by MPS (recommended working set)
    ("rtx8", 8 * GB),
    ("rtx12", 12 * GB),
    ("m2", int(16 * GB * 2 / 3)),
]
BATCH, MICRO = 8, 2
SIZES = [  # (blocks, hidden) of the tiny Transformer family
    (2, 128), (4, 192), (4, 256), (6, 256), (6, 320), (6, 384), (8, 384), (8, 448), (8, 512), (10, 512),
    (12, 512), (12, 576), (14, 576), (14, 640), (16, 640), (16, 704), (18, 704), (20, 768),
]


def model_cfg(blocks: int, hidden: int) -> dict:
    return {"type": "tiny_transformer", "layers": blocks, "hidden_size": hidden, "heads": 8, "vocab_size": 1024,
            "seq_len": 64}


def _opts(num_stages=None) -> PlannerOptions:
    return PlannerOptions(optimizer="adamw", num_microbatches=BATCH // MICRO, headroom_fraction=0.10,
                          headroom_min_bytes=8 * MB, num_stages=num_stages)


def _train(cfg: dict, plan, threads: dict[str, int], steps: int) -> dict:
    stages = [LocalStage((s.start, s.end), "cpu", threads=threads[s.worker_id]) for s in plan.stages]
    settings = PipelineSettings("capacity", steps=steps, batch_size=BATCH, num_microbatches=BATCH // MICRO,
                                log_every=1000, timeout_s=300)
    t0 = time.perf_counter()
    res = run_local_pipeline(cfg, stages, settings, optimizer="adamw", lr=3e-4, timeout_s=3600)
    wall = time.perf_counter() - t0
    step_s = statistics.median(m["step_s"] for m in res[0].step_metrics[1:])
    held = []
    for r, s in zip(res, plan.stages):
        # Optimizer state only exists after the first update: use the last step, and the
        # peak saved activations over all steps.
        m = r.step_metrics[-1]["memory"]
        peak = max(x["memory"]["saved_activations_peak"] for x in r.step_metrics)
        h = m["parameters"] + m["gradients"] + m["optimizer_state"] + peak
        held.append((s.worker_name, h, s.memory.budget))
    return {"steps_per_s": 1 / step_s, "step_s": step_s, "held": held, "wall_s": wall,
            "loss_first": res[0].losses[0], "loss_last": res[0].losses[-1]}


def run_emulated(steps: int = 5, write_doc: bool = True) -> str:
    workers = [EmulatedWorker(n, threads=1, memory_budget_bytes=mem // SCALE, label=f"{mem / GB:.1f} GB / {SCALE}")
               for n, mem in TARGET]
    profiles = profile_emulated(workers)
    threads = {w.name: w.threads for w in workers}
    rows, single_max, mesh_max = [], {w.name: None for w in workers}, None
    for blocks, hidden in SIZES:
        cfg = model_cfg(blocks, hidden)
        spec = build_model_spec(cfg)
        layers = profile_model(spec, MICRO)
        P = spec.parameter_count()
        cells = []
        for p in profiles:
            plan = make_plan("auto", layers, [p], NetworkModel(), _opts(num_stages=1))
            cells.append("fits" if plan.feasible else "-")
            if plan.feasible:
                single_max[p.name] = (P, cfg, plan)
        mesh = make_plan("auto", layers, profiles, NetworkModel(), _opts())
        need = sum(s.memory.required for s in mesh.stages) if mesh.stages else None
        if mesh.feasible:
            mesh_max = (P, cfg, mesh)
        rows.append([f"{blocks}x{hidden}", f"{P / 1e6:.1f}M", *cells,
                     f"fits ({len(mesh.stages)} stages)" if mesh.feasible else "-",
                     f"{need / MB:.0f} MB" if need else "-"])
        if not mesh.feasible and all(c == "-" for c in cells):
            break  # larger sizes cannot fit either

    best_single_name, best_single = max(((n, v) for n, v in single_max.items() if v), key=lambda kv: kv[1][0])
    P_single, single_cfg, single_plan = best_single
    P_mesh, mesh_cfg, mesh_plan = mesh_max
    run_single = _train(single_cfg, single_plan, threads, steps)
    run_mesh = _train(mesh_cfg, mesh_plan, threads, steps)
    # Same model as the single-device maximum, but on the mesh: apples-to-apples overhead.
    same_layers = profile_model(build_model_spec(single_cfg), MICRO)
    same_plan = make_plan("auto", same_layers, profiles, NetworkModel(), _opts(num_stages=len(profiles)))
    run_same = _train(single_cfg, same_plan, threads, steps) if same_plan.feasible else None

    gain = P_mesh / P_single
    penalty = run_single["steps_per_s"] / run_mesh["steps_per_s"]

    def held(run):
        return ", ".join(f"{n}: {h / MB:.0f}/{b / MB:.0f} MB" for n, h, b in run["held"])

    budget_rows = [[w.name, f"{mem / GB:.1f} GB", f"{w.memory_budget_bytes / MB:.0f} MB",
                    f"{p.measured_flops / 1e9:.1f}"] for (n, mem), w, p in zip(TARGET, workers, profiles)]
    md = "\n\n".join([
        environment_line(),
        f"**Mode: emulated.** Target cluster memory scaled down {SCALE}x and emulated by single-threaded CPU "
        "processes on one host; memory limits are the planner's accounting (params + grads + AdamW state + "
        "saved activations + temporary, 10% headroom) against the emulated budget, *not* real device OOMs. The "
        "largest models were then actually trained and the tensor bytes each stage held were measured. "
        f"Model family: tiny Transformer (vocab 1024, seq 64, 8 heads), AdamW, batch {BATCH}, "
        f"{BATCH // MICRO} microbatches. Parameter counts are as built (not scaled).",
        md_table(["worker", "real memory", "emulated budget", "measured GFLOP/s"], budget_rows),
        md_table(["model (blocks x hidden)", "params", *[f"{w.name} alone" for w in workers], "mesh (auto)",
                  "mesh total required"], rows),
        md_table(["", "P_max", "device(s)", "steps/s (measured)", "tensor MB held/budget", "loss (first -> last)"], [
            ["single device", f"{P_single / 1e6:.1f}M", best_single_name, f"{run_single['steps_per_s']:.3f}",
             held(run_single), f"{run_single['loss_first']:.3f} -> {run_single['loss_last']:.3f}"],
            ["MeshTrain", f"{P_mesh / 1e6:.1f}M", " -> ".join(s.worker_name for s in mesh_plan.stages),
             f"{run_mesh['steps_per_s']:.3f}", held(run_mesh),
             f"{run_mesh['loss_first']:.3f} -> {run_mesh['loss_last']:.3f}"],
        ] + ([["MeshTrain, same model as single", f"{P_single / 1e6:.1f}M",
               " -> ".join(s.worker_name for s in same_plan.stages), f"{run_same['steps_per_s']:.3f}",
               held(run_same), f"{run_same['loss_first']:.3f} -> {run_same['loss_last']:.3f}"]] if run_same else [])),
        f"**capacity_gain = P_mesh_max / max(P_single_max) = {P_mesh / 1e6:.1f}M / {P_single / 1e6:.1f}M = "
        f"{gain:.2f}x**  \n**throughput penalty = {run_single['steps_per_s']:.3f} / {run_mesh['steps_per_s']:.3f} "
        f"steps/s = {penalty:.2f}x slower**"
        + (f"  \nSame model ({P_single / 1e6:.1f}M) alone vs on the mesh: {run_single['steps_per_s']:.3f} vs "
           f"{run_same['steps_per_s']:.3f} steps/s" if run_same else ""),
        "_Caveat: these throughput numbers are **not representative of a real cluster**. Each emulated worker "
        "is a single-threaded process, so the 3-stage mesh uses 3 CPU cores where the single device uses 1, "
        "and links are loopback TCP (no real network cost). On real GPUs over a LAN the mesh is expected to be "
        "much slower than a single device; only the capacity ratio is meaningful here._",
        f"_{steps} steps per run; loss changes over so few steps are not a convergence result. "
        "Real-hardware capacity numbers require `meshtrain experiment capacity --mode hardware` on the "
        "physical cluster; none have been recorded yet._",
    ])
    if write_doc:
        update_section("experiment5-emulated", md)
    return md


def run_hardware(client, sizes=SIZES, steps: int = 5, write_doc: bool = True, poll_s: float = 1.0) -> str:
    """Real-device capacity search, with the mesh constrained to use every online worker."""
    workers = [w for w in client.status()["workers"] if w["status"] == "ONLINE"]
    if len(workers) < 2:
        raise ValueError("hardware capacity comparison requires at least two online workers; check meshtrain status")

    def submit(cfg_model, *, require=None, num_stages=None, strategy="auto", enforce=False):
        cfg = {"job": {"name": "capacity"}, "model": cfg_model,
               "training": {"batch_size": BATCH, "microbatch_size": MICRO, "learning_rate": 3e-4,
                            "optimizer": "adamw", "steps": steps, "log_every": 1},
               "placement": {"strategy": strategy, "num_stages": num_stages, "enforce_memory_check": enforce},
               "workers": {"allow": ["cuda", "mps", "cpu"], "require": require or []},
               "network": {"timeout_s": 300}}
        if require:
            n = build_model_spec(cfg_model).num_layers
            cfg["placement"] = {"strategy": "manual", "enforce_memory_check": False,
                                "stages": [{"worker": require[0], "layers": [0, n]}]}
        try:
            job = client.start_job(cfg)
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200]}
        while True:
            j = client.job(job["job_id"])
            if j["status"] != "RUNNING":
                break
            time.sleep(poll_s)
        ok = j["status"] == "COMPLETED"
        sps = j["summary"]["steps_per_s"] if ok and j.get("summary") else None
        return {"ok": ok, "error": j.get("error"), "steps_per_s": sps, "stages": len(j["stage_workers"])}

    results = {w["worker_id"]: {} for w in workers}
    mesh = {}
    for blocks, hidden in sizes:
        cfg = model_cfg(blocks, hidden)
        P = build_model_spec(cfg).parameter_count()
        for w in workers:
            if results[w["worker_id"]].get("failed"):
                continue
            r = submit(cfg, require=[w["worker_id"]])
            if r["ok"]:
                results[w["worker_id"]].update(max_P=P, steps_per_s=r["steps_per_s"])
            else:
                results[w["worker_id"]]["failed"] = r["error"]
        if not mesh.get("failed"):
            r = submit(cfg, num_stages=len(workers), enforce=True)
            if r["ok"]:
                mesh.update(max_P=P, steps_per_s=r["steps_per_s"], stages=r["stages"])
            else:
                mesh["failed"] = r["error"]
        if mesh.get("failed") and all(v.get("failed") for v in results.values()):
            break
    best = max((v for v in results.values() if "max_P" in v), key=lambda v: v["max_P"], default=None)
    def clean(err):
        return (err or "").replace("\n", " ").replace("|", "/")[:120]

    rows = [[wid, f"{v.get('max_P', 0) / 1e6:.1f}M", f"{v.get('steps_per_s') or 0:.3f}",
             1 if "max_P" in v else "-", clean(v.get("failed"))]
            for wid, v in results.items()]
    rows.append(["MeshTrain", f"{mesh.get('max_P', 0) / 1e6:.1f}M", f"{mesh.get('steps_per_s') or 0:.3f}",
                 mesh.get("stages", "-"), clean(mesh.get("failed"))])
    md = [environment_line(), "**Mode: hardware** (real devices via the coordinator). "
          f"Mesh trials request {len(workers)} stages, one per online worker. "
          "P_max is the largest successful size tested; a zero means no tested size succeeded, not zero capacity.",
          md_table(["device", "P_max", "steps/s at P_max", "stages at P_max", "first failure"], rows)]
    if best and mesh.get("max_P"):
        md.append(f"capacity_gain = {mesh['max_P'] / best['max_P']:.2f}x, throughput penalty = "
                  f"{(best['steps_per_s'] or 0) / (mesh['steps_per_s'] or 1):.2f}x")
    text = "\n\n".join(md)
    if write_doc:
        update_section("experiment5-hardware", text)
    return text


def run_experiment5(mode: str = "emulated", cluster_file: str | None = None, steps: int = 5, *, client=None) -> str:
    """``client``: a ControlClient for the running cluster (hardware mode; defaults to the remembered cluster)."""
    if mode == "emulated":
        return run_emulated(steps=steps)
    if client is None:
        from meshtrain.cli import _client

        client = _client()
    with client.http:
        return run_hardware(client, steps=steps)

"""Experiment 4: placement strategies (equal vs compute-aware vs memory+compute+network-aware).

With ``--cluster bench.json`` (output of ``meshtrain cluster benchmark
--output``) the plans are computed for that real cluster (predictions only).
Without it, the experiment runs on emulated CPU workers and *measures* each
feasible plan by actually training.
"""

from __future__ import annotations

import json
import statistics

from meshtrain.experiments.emulation import MB, EmulatedWorker, profile_emulated
from meshtrain.experiments.results_doc import environment_line, md_table, update_section
from meshtrain.models import build_model_spec
from meshtrain.planner.cost import NetworkModel
from meshtrain.planner.graph import profile_model
from meshtrain.planner.partition import PlannerOptions, WorkerProfile, make_plan
from meshtrain.runtime.local import LocalStage, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings

MODEL = {"type": "tiny_transformer", "layers": 10, "hidden_size": 256, "heads": 4, "vocab_size": 512,
         "seq_len": 64}
BATCH, MICRO = 16, 4
WORKERS = [
    EmulatedWorker("fast-small", threads=2, memory_budget_bytes=90 * MB, label="2 threads, 90 MB"),
    EmulatedWorker("slow-large", threads=1, memory_budget_bytes=260 * MB, label="1 thread, 260 MB"),
    EmulatedWorker("slow-mid", threads=1, memory_budget_bytes=150 * MB, label="1 thread, 150 MB"),
]


def _opts(n_mb: int, num_stages: int | None = None) -> PlannerOptions:
    return PlannerOptions(optimizer="adamw", num_microbatches=n_mb, headroom_fraction=0.1,
                          headroom_min_bytes=4 * MB, num_stages=num_stages)


def _measure(plan, workers_by_name: dict[str, EmulatedWorker], steps: int) -> dict:
    stages = [LocalStage((s.start, s.end), "cpu", threads=workers_by_name[s.worker_id].threads) for s in plan.stages]
    settings = PipelineSettings("placement", steps=steps, batch_size=BATCH, num_microbatches=BATCH // MICRO,
                                log_every=1000)
    results = run_local_pipeline(MODEL, stages, settings, optimizer="adamw", lr=3e-4)
    step_s = statistics.median(r["step_s"] for r in results[0].step_metrics[2:])
    held = {}
    for r in results:
        m = r.step_metrics[-1]["memory"]
        held[r.stage_index] = m["parameters"] + m["gradients"] + m["optimizer_state"] + m["saved_activations_peak"]
    return {"step_s": step_s, "held": held, "loss_first": results[0].losses[0], "loss_last": results[0].losses[-1]}


def run_experiment4(cluster_file: str | None = None, steps: int = 12, write_doc: bool = True) -> str:
    spec = build_model_spec(MODEL)
    layers = profile_model(spec, MICRO)
    n_mb = BATCH // MICRO
    sections = [environment_line()]
    if cluster_file:
        with open(cluster_file) as f:
            bench = json.load(f)
        profiles = [WorkerProfile(w, w, r["backend"], int(r.get("memory_total", 8 * 1024**3)), r["measured_flops"])
                    for w, r in bench["workers"].items()]
        network = NetworkModel.from_measurements(bench.get("links", {}))
        rows = []
        for strat in ("equal", "compute", "auto"):
            p = make_plan(strat, layers, profiles, network, _opts(n_mb))
            rows.append([strat, " / ".join(f"{s.worker_name}:{s.start}-{s.end - 1}" for s in p.stages),
                         "yes" if p.feasible else "NO", f"{p.predicted_step_s * 1000:.1f}",
                         f"{p.communication_bytes_per_step / 1e6:.2f}"])
        sections += [f"Cluster from `{cluster_file}` (predictions only).",
                     md_table(["strategy", "placement", "feasible", "predicted ms/step", "MB/step"], rows)]
    else:
        profiles = profile_emulated(WORKERS)
        by_name = {w.name: w for w in WORKERS}
        sections.append(
            "Emulated heterogeneous CPU workers on one host (compute differences are real: separate processes "
            "with different torch thread counts, measured by the benchmark; memory budgets are emulated and "
            "enforced by the planner). Model: tiny Transformer "
            f"({spec.parameter_count() / 1e6:.2f}M params, {spec.num_layers} layers), AdamW, batch {BATCH}, "
            f"{n_mb} microbatches, {steps} steps per plan.")
        sections.append(md_table(["worker", "emulated as", "measured GFLOP/s", "compute_score"],
                                 [[p.name, by_name[p.name].label, f"{p.measured_flops / 1e9:.1f}",
                                   f"{p.compute_score:.2f}"] for p in profiles]))
        rows = []
        for strat in ("equal", "compute", "auto"):
            p = make_plan(strat, layers, profiles, NetworkModel(), _opts(n_mb, num_stages=3 if strat != "auto" else None))
            placement = " / ".join(f"{s.worker_name}:{s.start}-{s.end - 1}" for s in p.stages)
            m = _measure(p, by_name, steps)  # infeasible plans run anyway: CPU cannot really OOM here
            within = all(m["held"][s.stage_index] <= s.memory.budget for s in p.stages)
            feas = "yes" if p.feasible else "NO: " + "; ".join(p.violations)
            rows.append([strat, placement, feas, f"{p.predicted_step_s * 1000:.1f}", f"{m['step_s'] * 1000:.1f}",
                         " / ".join(f"{m['held'][s.stage_index] / MB:.1f}/{s.memory.budget / MB:.0f}" for s in p.stages)
                         + (" ok" if within else " EXCEEDED"),
                         f"{m['loss_first']:.3f} -> {m['loss_last']:.3f}"])
        sections.append(md_table(["strategy", "placement (worker:layers)", "memory-feasible", "predicted ms/step",
                                  "measured ms/step", "tensor MB held/budget per stage", "loss"], rows))
        sections.append("_Plans that violate an emulated memory budget were still executed (a CPU process cannot "
                        "hit the emulated limit) so their speed can be compared; on a real accelerator they would "
                        "be rejected by the planner or fail with out-of-memory._")
        sections.append("_Held memory = parameters + gradients + optimizer state + peak saved activations (boundary "
                        "tensors + tensors autograd saved for backward), measured from what each stage actually "
                        "held. Network links are loopback (not emulated), so "
                        "the network-aware part of `auto` has little to optimise here._")
    md = "\n\n".join(sections)
    if write_doc:
        update_section("experiment4" + ("-cluster" if cluster_file else "-emulated"), md)
    return md

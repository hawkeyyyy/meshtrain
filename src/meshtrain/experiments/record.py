"""Record a finished cluster run (runs/<job-id>) into docs/v1-results.md.

Used for Experiments 2 and 3 (CUDA + CUDA, CUDA + CUDA + MPS):

    meshtrain train configs/two_cuda.yaml
    meshtrain results record runs/<job-id> --section experiment2

Correctness on real hardware is checked by replaying the first steps of the
same config in a single CPU process and comparing loss curves (CUDA/MPS
kernels differ numerically from CPU, so a small deviation is expected).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from meshtrain.config import parse_config
from meshtrain.experiments.results_doc import md_table, update_section
from meshtrain.runtime.local import reference_step
from meshtrain.runtime.pipeline import PipelineSettings
from meshtrain.telemetry import read_jsonl


def record_run(run_dir: str, section: str, reference_steps: int = 20, cluster_status: dict | None = None,
               write_doc: bool = True) -> str:
    run = Path(run_dir)
    cfg = parse_config(yaml.safe_load((run / "config.yaml").read_text()))
    plan = json.loads((run / "plan.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    records = read_jsonl(run / "metrics.jsonl")
    losses = [r["loss"] for r in sorted((r for r in records if "loss" in r), key=lambda r: r["step"])]
    n = min(reference_steps, len(losses))
    settings = PipelineSettings("ref", steps=n, batch_size=cfg.training.batch_size,
                                num_microbatches=cfg.training.num_microbatches)
    ref, _, _ = reference_step(cfg.model.spec_kwargs(), settings, seed=cfg.job.seed, optimizer=cfg.training.optimizer,
                               lr=cfg.training.learning_rate, steps=n)
    dev = max(abs(a - b) / max(abs(b), 1e-8) for a, b in zip(losses[:n], ref))
    spec = cfg.build_model_spec()
    hw = []
    workers = {w["worker_id"]: w for w in (cluster_status or {}).get("workers", [])}
    for s in plan["stages"]:
        w = workers.get(s["worker_id"], {})
        dev_name = (w.get("device") or {}).get("name", s["backend"])
        hw.append(f"{s['worker']} ({s['backend']}, {dev_name})")
    stage_rows = []
    for s in plan["stages"]:
        st = summary["stages"].get(str(s["stage"]), {})
        mem = st.get("memory", {})
        stage_rows.append([
            s["stage"], s["worker"], s["backend"], f"{s['layers'][0]}-{s['layers'][1] - 1}",
            f"{s['memory']['required'] / 1e9:.2f} / {s['memory']['budget'] / 1e9:.2f}",
            f"{mem.get('device_peak', 0) / 1e9:.2f}",
            f"{st.get('forward_s', 0) * 1e3:.1f}", f"{st.get('backward_s', 0) * 1e3:.1f}",
            f"{st.get('comm_s', 0) * 1e3:.1f}", f"{st.get('idle_s', 0) * 1e3:.1f}",
            f"{st.get('bytes_sent_per_step', 0) / 1e6:.2f}",
        ])
    md = "\n\n".join([
        f"Run `{run.name}` — hardware: {' -> '.join(hw)}",
        f"Model: {cfg.model.type} {spec.parameter_count() / 1e6:.1f}M params ({spec.num_layers} layers); batch "
        f"{cfg.training.batch_size}, {cfg.training.num_microbatches} microbatches, {cfg.training.optimizer}.",
        md_table(["steps", "loss first -> last", "ms/step", "steps/s", "samples/s", "MB/step on network",
                  f"max rel. loss deviation vs CPU single-process (first {n} steps)"],
                 [[summary["steps"], f"{summary['initial_loss']:.4f} -> {summary['final_loss']:.4f}",
                   f"{summary['mean_step_s'] * 1e3:.1f}", f"{summary['steps_per_s']:.2f}",
                   f"{summary['samples_per_s']:.1f}", f"{summary['bytes_per_step'] / 1e6:.2f}", f"{dev:.2e}"]]),
        "_device peak = torch allocator peak (CUDA), driver allocation (MPS), process RSS (CPU)._",
        md_table(["stage", "worker", "backend", "layers", "planned GB (need / usable)", "device peak GB",
                  "fwd ms", "bwd ms", "comm ms", "idle ms", "MB sent/step"], stage_rows),
    ])
    if write_doc:
        update_section(section, md)
    return md

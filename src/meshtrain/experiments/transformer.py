"""Milestone 8: tiny Transformer benchmark (3 stages, microbatched, TCP).

Checks gradient equivalence for one step, then trains and compares the loss
trajectory with single-process training on the same data.
"""

from __future__ import annotations

from meshtrain.experiments.correctness import run_gradient_equivalence
from meshtrain.experiments.results_doc import environment_line, md_table, update_section
from meshtrain.models import build_model_spec
from meshtrain.runtime.local import LocalStage, reference_step, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings
from meshtrain.summary import summarize

MODEL = {"type": "tiny_transformer", "layers": 6, "hidden_size": 128, "heads": 4, "vocab_size": 128, "seq_len": 32}
STAGES = [LocalStage((0, 3)), LocalStage((3, 5)), LocalStage((5, 8))]


def run_experiment_transformer(steps: int = 150, write_doc: bool = True) -> str:
    spec = build_model_spec(MODEL)
    rows, _ = run_gradient_equivalence(MODEL, STAGES, batch_size=16, num_microbatches=4, transport="tcp")
    worst = max(rows, key=lambda r: r.relative)
    settings = PipelineSettings("transformer", steps=steps, batch_size=16, num_microbatches=4, log_every=1000)
    res = run_local_pipeline(MODEL, STAGES, settings, optimizer="adamw", lr=1e-3, transport="tcp",
                             capture_params=True)
    ref_losses, _, _ = reference_step(MODEL, settings, optimizer="adamw", lr=1e-3, steps=steps)
    losses = res[0].losses
    dev = max(abs(a - b) / max(abs(b), 1e-8) for a, b in zip(losses, ref_losses))
    s = summarize([m for r in res for m in r.step_metrics])
    changed = all(all(not (r.initial_params[k] == r.final_params[k]).all() for k in r.initial_params) for r in res)
    curve = [[i, f"{losses[i]:.4f}", f"{ref_losses[i]:.4f}"] for i in sorted({0, 25, 50, 100, steps - 1}) if i < steps]
    stage_rows = [[i, st["worker"], f"{st['forward_s'] * 1e3:.1f}", f"{st['backward_s'] * 1e3:.1f}",
                   f"{st['comm_s'] * 1e3:.1f}", f"{st['idle_s'] * 1e3:.1f}", f"{st['bytes_sent_per_step'] / 1e6:.3f}",
                   f"{(st['memory'].get('parameters', 0)) / 1e6:.2f}",
                   f"{st['memory'].get('saved_activations_peak', 0) / 1e6:.2f}"]
                  for i, st in s["stages"].items()]
    md = "\n\n".join([
        environment_line(),
        f"CPU only, 3 stage processes over loopback TCP. Model: {spec.parameter_count() / 1e6:.2f}M params "
        f"({spec.num_layers} layers: embedding, 6 blocks, LM head), stages {[s.layers for s in STAGES]}, "
        "batch 16 = 4 microbatches of 4, AdamW lr 1e-3, synthetic arithmetic-sequence next-token task.",
        f"- One-step gradient equivalence: max relative error {worst.relative:.2e} ({worst.name}), "
        f"max abs error {max(r.max_abs for r in rows):.2e} over {len(rows)} parameter tensors",
        f"- Training: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {steps} steps; parameters changed on every "
        f"stage: {changed}; max relative deviation from single-process loss curve: {dev:.2e}",
        f"- Throughput: {s['mean_step_s'] * 1e3:.1f} ms/step, {s['samples_per_s']:.1f} samples/s, "
        f"{s['bytes_per_step'] / 1e6:.3f} MB sent per step",
        md_table(["step", "MeshTrain loss", "single-process loss"], curve),
        md_table(["stage", "worker", "fwd ms", "bwd ms", "comm ms", "idle ms", "MB sent/step", "params MB",
                  "saved act peak MB"], stage_rows),
    ])
    if write_doc:
        update_section("milestone8-transformer", md)
    return md

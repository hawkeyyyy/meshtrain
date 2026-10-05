"""Experiment 1: gradient equivalence between single-process and distributed execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from meshtrain.runtime.local import LocalStage, global_param_names, reference_step, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings


@dataclass
class GradientError:
    name: str
    stage: int
    max_abs: float
    mean_abs: float
    relative: float  # ||g_dist - g_ref|| / ||g_ref||


def compare_gradients(reference: dict[str, torch.Tensor], distributed: dict[str, torch.Tensor],
                      stage_of: dict[str, int] | None = None) -> list[GradientError]:
    if set(reference) != set(distributed):
        missing = set(reference) ^ set(distributed)
        raise AssertionError(f"gradient sets differ: {sorted(missing)}")
    rows = []
    for name in reference:
        r = reference[name].double()
        d = distributed[name].double().to(r.device)
        diff = (d - r).abs()
        denom = r.norm().item()
        rows.append(GradientError(
            name=name,
            stage=(stage_of or {}).get(name, -1),
            max_abs=diff.max().item(),
            mean_abs=diff.mean().item(),
            relative=(d - r).norm().item() / denom if denom > 0 else diff.max().item(),
        ))
    return rows


def format_gradient_report(rows: list[GradientError]) -> str:
    lines = [f"{'Parameter':<28}{'stage':>6}{'max_abs':>12}{'mean_abs':>12}{'relative':>12}", "-" * 70]
    for r in rows:
        lines.append(f"{r.name:<28}{r.stage:>6}{r.max_abs:>12.2e}{r.mean_abs:>12.2e}{r.relative:>12.2e}")
    return "\n".join(lines)


def run_gradient_equivalence(model_cfg: dict, stages: list[LocalStage], *, batch_size: int = 32,
                             num_microbatches: int = 1, seed: int = 0, transport: str = "pipe",
                             reference_device: str = "cpu") -> tuple[list[GradientError], dict]:
    """One training step, single-process vs. distributed; returns per-parameter errors."""
    settings = PipelineSettings(job_id="grad-equivalence", steps=1, batch_size=batch_size,
                                num_microbatches=num_microbatches, capture_gradients_at_step=0, log_every=1)
    ref_losses, ref_grads, _ = reference_step(model_cfg, settings, seed=seed, device=reference_device)
    results = run_local_pipeline(model_cfg, stages, settings, seed=seed, transport=transport)
    dist_grads, stage_of = {}, {}
    for r in results:
        g = global_param_names(stages, r.gradients, r.stage_index)
        dist_grads.update(g)
        stage_of.update({k: r.stage_index for k in g})
    rows = compare_gradients(ref_grads, dist_grads, stage_of)
    info = {"reference_loss": ref_losses[0], "distributed_loss": results[0].losses[0],
            "stages": [list(s.layers) for s in stages], "devices": [s.device for s in stages],
            "transport": transport, "microbatches": num_microbatches}
    return rows, info


def run_experiment1(transport: str = "pipe", write_doc: bool = True) -> str:
    """Experiment 1: CPU single process vs distributed CPU stages; returns markdown."""
    from meshtrain.experiments.results_doc import environment_line, md_table, update_section

    cases = [
        ("MLP 2 stages", {"type": "mlp"}, [LocalStage((0, 3)), LocalStage((3, 7))], 32, 1),
        ("MLP 3 stages", {"type": "mlp"}, [LocalStage((0, 2)), LocalStage((2, 5)), LocalStage((5, 7))], 32, 1),
        ("MLP 2 stages, 4 microbatches", {"type": "mlp"}, [LocalStage((0, 3)), LocalStage((3, 7))], 32, 4),
        ("Transformer (4 blocks, d=64) 3 stages, 2 microbatches",
         {"type": "tiny_transformer", "layers": 4, "hidden_size": 64, "heads": 4, "vocab_size": 64, "seq_len": 16},
         [LocalStage((0, 2)), LocalStage((2, 4)), LocalStage((4, 6))], 8, 2),
    ]
    summary_rows, details = [], []
    for title, cfg, stages, bs, mbs in cases:
        rows, info = run_gradient_equivalence(cfg, stages, batch_size=bs, num_microbatches=mbs, transport=transport)
        worst = max(rows, key=lambda r: r.relative)
        summary_rows.append([title, transport, len(stages), mbs, len(rows), f"{max(r.max_abs for r in rows):.2e}",
                             f"{worst.relative:.2e}", f"{abs(info['reference_loss'] - info['distributed_loss']):.2e}",
                             "PASS" if worst.relative < 1e-4 else "FAIL"])
        per_stage = {}
        for r in rows:
            per_stage.setdefault(r.stage, []).append(r)
        details.append(f"**{title}** — per-stage max abs error: " + ", ".join(
            f"stage{s}: {max(x.max_abs for x in v):.2e}" for s, v in sorted(per_stage.items())))
    md = "\n\n".join([
        environment_line(),
        "Device: CPU only (each stage in its own OS process; tensors cross process boundaries as serialized "
        "TensorPackets). Batch split identically in the single-process reference.",
        md_table(["case", "transport", "stages", "microbatches", "params", "max abs err", "max relative err",
                  "loss abs diff", "result"], summary_rows),
        "\n".join(f"- {d}" for d in details),
    ])
    if write_doc:
        update_section(f"experiment1-{transport}", md)
    return md

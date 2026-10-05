"""Heterogeneous-device correctness: a pipeline on real devices vs a CPU reference.

    meshtrain experiment device-correctness --devices cuda,cuda,mps

Compares, for the same seeds and data:

* loss per step                         (relative error)
* first-step gradients, every parameter (relative L2 error)
* per-parameter gradient norms per step (relative error)
* first-update parameter deltas         (relative error of ||Δθ||)
* output logits of the final weights on a probe batch (max abs / relative error)

Tolerances (documented, not bitwise): CPU-only pipelines 1e-5 relative;
pipelines with any CUDA/MPS stage 5e-3 relative (different kernels and
reduction orders; MPS also lacks float64). The comparison runs every stage in
its own process on this machine; across machines use ``meshtrain results
record``, which replays the loss curve on CPU.
"""

from __future__ import annotations

import torch

from meshtrain.experiments.correctness import compare_gradients
from meshtrain.models import build_model_spec
from meshtrain.runtime.local import LocalStage, global_param_names, run_local_pipeline
from meshtrain.runtime.pipeline import PipelineSettings
from meshtrain.runtime.stage import build_optimizer

CPU_TOL = 1e-5
ACCEL_TOL = 5e-3

DEFAULT_MODEL = {"type": "tiny_transformer", "layers": 4, "hidden_size": 128, "heads": 4, "vocab_size": 256,
                 "seq_len": 32}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def _reference(model_cfg, settings, optimizer, lr, steps):
    spec = build_model_spec(model_cfg)
    model = spec.build_full()
    opt = build_optimizer(optimizer, model.parameters(), lr)
    M = settings.num_microbatches
    losses, norms, first_grads, deltas = [], [], None, None
    for step in range(steps):
        x, y = spec.make_batch(step, settings.batch_size)
        total = 0.0
        for xm, ym in zip(x.chunk(M), y.chunk(M)):
            loss = spec.loss_fn(model(xm), ym)
            (loss / M).backward()
            total += loss.item()
        losses.append(total / M)
        norms.append({n: float(p.grad.norm()) for n, p in model.named_parameters()})
        if step == 0:
            first_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            deltas = {n: float((p.detach() - before[n]).norm()) for n, p in model.named_parameters()}
    return spec, model, losses, norms, first_grads, deltas


def run_device_correctness(devices: list[str], model_cfg: dict | None = None, steps: int = 3,
                           batch_size: int = 8, num_microbatches: int = 4, optimizer: str = "adamw",
                           lr: float = 1e-3, schedule: str = "1f1b") -> dict:
    model_cfg = model_cfg or DEFAULT_MODEL
    n_layers = build_model_spec(model_cfg).num_layers
    k = len(devices)
    stages = [LocalStage((n_layers * i // k, n_layers * (i + 1) // k), d) for i, d in enumerate(devices)]
    settings = PipelineSettings("device-correctness", steps=steps, batch_size=batch_size,
                                num_microbatches=num_microbatches, schedule=schedule, async_transport=True,
                                capture_gradients_at_step=0, correctness_probe_steps=steps)
    res = run_local_pipeline(model_cfg, stages, settings, optimizer=optimizer, lr=lr, transport="tcp",
                             capture_params=True)
    spec, ref_model, ref_losses, ref_norms, ref_grads, ref_deltas = _reference(model_cfg, settings, optimizer,
                                                                                lr, steps)
    grads, final, norms, deltas = {}, {}, [dict() for _ in range(steps)], {}
    for r in res:
        grads.update(global_param_names(stages, r.gradients, r.stage_index))
        final.update(global_param_names(stages, r.final_params, r.stage_index))
        for i, m in enumerate(r.step_metrics[:steps]):
            c = m["correctness"]
            norms[i].update(global_param_names(stages, {k2: torch.tensor(v) for k2, v in c["grad_norms"].items()},
                                               r.stage_index))
            if "param_delta_norms" in c:
                deltas.update(global_param_names(
                    stages, {k2: torch.tensor(v) for k2, v in c["param_delta_norms"].items()}, r.stage_index))
    loss_err = max(_rel(a, b) for a, b in zip(res[0].losses, ref_losses))
    grad_err = max(r.relative for r in compare_gradients(ref_grads, grads))
    norm_err = max(_rel(float(norms[i][n]), ref_norms[i][n]) for i in range(steps) for n in ref_norms[i])
    delta_err = max(_rel(float(deltas[n]), ref_deltas[n]) for n in ref_deltas if ref_deltas[n] > 0)
    dist_model = spec.build_full()
    dist_model.load_state_dict({k2: v for k2, v in final.items()})
    xp, _ = spec.make_batch(10_000, batch_size)
    with torch.no_grad():
        out_d, out_r = dist_model(xp), ref_model(xp)
    out_abs = float((out_d - out_r).abs().max())
    out_rel = float((out_d - out_r).norm() / out_r.norm())
    tol = CPU_TOL if all(d == "cpu" for d in devices) else ACCEL_TOL
    checks = {"loss": loss_err, "gradients": grad_err, "grad_norms": norm_err, "param_deltas": delta_err,
              "outputs": out_rel}
    return {"devices": devices, "stages": [list(s.layers) for s in stages], "steps": steps, "tolerance": tol,
            "errors": checks, "output_max_abs": out_abs, "passed": all(v <= tol for v in checks.values()),
            "losses": res[0].losses, "reference_losses": ref_losses}


def format_device_correctness(r: dict) -> str:
    lines = [f"devices: {' -> '.join(r['devices'])}   stages {r['stages']}   tolerance {r['tolerance']:.0e}"]
    for k, v in r["errors"].items():
        lines.append(f"  {k:<14} max relative error {v:.2e}  {'ok' if v <= r['tolerance'] else 'FAIL'}")
    lines.append(f"  output max abs error {r['output_max_abs']:.2e}")
    lines.append(f"  loss {r['losses'][0]:.4f} -> {r['losses'][-1]:.4f} (reference {r['reference_losses'][0]:.4f} "
                 f"-> {r['reference_losses'][-1]:.4f})")
    lines.append("PASS" if r["passed"] else "FAIL")
    return "\n".join(lines)

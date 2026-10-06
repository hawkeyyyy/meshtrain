"""V2 correctness: static (V1.5) training vs offloaded training.

Both runs use the same model, seed, data, optimizer and microbatching on one
device (single stage, no network). After *every* optimizer step we compare:

* loss
* gradients (accumulated over the step's microbatches, before the update)
* parameters after the update (optimizer-state bugs show up here first)
* logits of a fixed probe batch

and report max / mean absolute difference per step. Optimizer bugs often look
like healthy training while silently diverging, hence the per-step check.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from meshtrain.models import build_model_spec
from meshtrain.runtime.offload import ResidencyPolicy
from meshtrain.runtime.pipeline import PipelineSettings, run_stage
from meshtrain.runtime.stage import Stage
from meshtrain.worker.device import select_device


def build_single_stage(model_cfg: dict, *, device: str = "cpu", policy: ResidencyPolicy | None = None,
                       budget_bytes: int | None = None, optimizer: str = "adamw", lr: float = 1e-3, seed: int = 0,
                       microbatch_size: int = 2):
    """One stage holding the whole model. ``auto_offload`` policies are resolved here."""
    spec = build_model_spec(model_cfg, seed=seed)
    adapter = select_device(device)
    if policy is not None and policy.strategy == "auto_offload" and policy.resident_groups is None:
        from meshtrain.planner.residency import plan_stage_residency

        plan = plan_stage_residency(spec, 0, spec.num_layers, microbatch_size=microbatch_size, budget=budget_bytes,
                                    policy=policy, optimizer=optimizer, backend=adapter.backend)
        policy = plan.apply(policy)
    stage = Stage(spec.build_full(), stage_index=0, num_stages=1, device=adapter, optimizer=optimizer, lr=lr,
                  loss_fn=spec.loss_fn, name="single", residency=policy, accelerator_budget=budget_bytes)
    return stage, spec


@dataclass
class Trace:
    losses: list[float]
    grads: list[dict[str, torch.Tensor]]
    params: list[dict[str, torch.Tensor]]
    logits: list[torch.Tensor]
    stats: list[dict]


def train_trace(model_cfg: dict, *, policy: ResidencyPolicy | None = None, device: str = "cpu",
                budget_bytes: int | None = None, steps: int = 3, batch_size: int = 8, microbatch_size: int = 2,
                optimizer: str = "adamw", lr: float = 1e-3, seed: int = 0, schedule: str = "1f1b",
                capture: bool = True) -> Trace:
    stage, spec = build_single_stage(model_cfg, device=device, policy=policy, budget_bytes=budget_bytes,
                                     optimizer=optimizer, lr=lr, seed=seed, microbatch_size=microbatch_size)
    probe_x, _ = spec.make_batch(10_000, microbatch_size)
    trace = Trace([], [], [], [], [])
    try:
        for i in range(steps):
            settings = PipelineSettings("v2-correctness", steps=1, batch_size=batch_size,
                                        num_microbatches=batch_size // microbatch_size, step_offset=i,
                                        log_every=10**6, schedule=schedule, trace=False,
                                        capture_gradients_at_step=i if capture else None)
            res = run_stage(stage, spec, settings, upstream=None, downstream=None, worker="single")
            trace.losses.append(res.losses[0])
            trace.stats.append(res.step_metrics[0])
            if capture:
                trace.grads.append(res.gradients)
                trace.params.append(stage.named_parameters_cpu())
                with torch.no_grad():
                    trace.logits.append(stage.module(stage.device.move_tensor(probe_x)).detach().float().cpu())
    finally:
        stage.close()
    return trace


def _diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    if set(a) != set(b):
        raise AssertionError(f"tensor sets differ: {sorted(set(a) ^ set(b))[:5]}")
    mx, total, n = 0.0, 0.0, 0
    for k in a:
        d = (a[k].double() - b[k].double()).abs()
        mx = max(mx, float(d.max()))
        total += float(d.sum())
        n += d.numel()
    return mx, total / max(n, 1)


def compare_traces(base: Trace, other: Trace) -> list[dict]:
    rows = []
    for i in range(len(base.losses)):
        g_max, g_mean = _diff(base.grads[i], other.grads[i])
        p_max, p_mean = _diff(base.params[i], other.params[i])
        l = (base.logits[i] - other.logits[i]).abs()
        rows.append({"step": i, "loss_base": base.losses[i], "loss_other": other.losses[i],
                     "loss_abs_diff": abs(base.losses[i] - other.losses[i]),
                     "grad_max_abs_diff": g_max, "grad_mean_abs_diff": g_mean,
                     "param_max_abs_diff": p_max, "param_mean_abs_diff": p_mean,
                     "logits_max_abs_diff": float(l.max()), "logits_mean_abs_diff": float(l.mean())})
    return rows


def format_rows(rows: list[dict]) -> str:
    head = f"{'step':>4} {'loss':>10} {'d_loss':>9} {'grad max/mean':>21} {'param max/mean':>21} {'logits max':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['step']:>4} {r['loss_base']:>10.5f} {r['loss_abs_diff']:>9.2e} "
                     f"{r['grad_max_abs_diff']:>10.2e}/{r['grad_mean_abs_diff']:<10.2e} "
                     f"{r['param_max_abs_diff']:>10.2e}/{r['param_mean_abs_diff']:<10.2e} "
                     f"{r['logits_max_abs_diff']:>10.2e}")
    return "\n".join(lines)


VARIANTS = {
    "manual_offload": ResidencyPolicy(strategy="manual_offload", prefetch_distance=0),
    "manual_offload+prefetch": ResidencyPolicy(strategy="manual_offload", prefetch_distance=1),
    "manual_offload+after_backward": ResidencyPolicy(strategy="manual_offload", eviction="after_backward"),
    "optimizer_offload": ResidencyPolicy(strategy="static", optimizer_execution="cpu_offload"),
    "manual_offload+optimizer_offload": ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"),
    "auto_offload": ResidencyPolicy(strategy="auto_offload", optimizer_execution="cpu_offload"),
}

DEFAULT_MODEL = {"type": "tiny_transformer", "layers": 4, "hidden_size": 64, "heads": 4, "vocab_size": 64,
                 "seq_len": 16}


def run_offload_correctness(model_cfg: dict | None = None, *, device: str = "cpu", steps: int = 4,
                            variants: list[str] | None = None, budget_mb: float | None = None) -> tuple[str, dict]:
    model_cfg = model_cfg or DEFAULT_MODEL
    budget = int(budget_mb * 1024**2) if budget_mb else None
    base = train_trace(model_cfg, device=device, steps=steps)
    out, text = {}, [f"V2 offload correctness on {device} -- {model_cfg} -- {steps} AdamW steps, 4 microbatches",
                     "baseline: memory.strategy static (V1.5 path)"]
    for name in variants or list(VARIANTS):
        pol = VARIANTS[name]
        tr = train_trace(model_cfg, policy=ResidencyPolicy(**vars(pol)), device=device, steps=steps,
                         budget_bytes=budget if pol.strategy == "auto_offload" else None)
        rows = compare_traces(base, tr)
        out[name] = rows
        text += ["", f"== {name}", format_rows(rows)]
    return "\n".join(text), out

"""Turns registered workers + a job config into a placement plan.

The coordinator never touches model weights or runs training; it only
profiles the model on the ``meta`` device and runs the planner.
"""

from __future__ import annotations

from meshtrain.config import MeshTrainConfig
from meshtrain.coordinator.state import WorkerRecord
from meshtrain.planner.cost import NetworkModel
from meshtrain.planner.graph import profile_model
from meshtrain.planner.partition import Plan, PlannerOptions, WorkerProfile, make_plan, plan_manual

# Used for workers that have not been benchmarked yet (documented, conservative).
UNBENCHMARKED_FLOPS = {"cuda": 5e12, "mps": 1e12, "cpu": 5e10}


def worker_profile(w: WorkerRecord) -> WorkerProfile:
    dev = w.device_info
    if w.backend == "cpu":
        total = int(w.hardware.get("ram_available") or w.hardware.get("ram_total", 0))
    else:
        total = int(dev.get("memory_total", 0))
    dtypes = ("float32", "float16", "bfloat16", "float64")
    if w.backend == "mps":
        dtypes = ("float32", "float16", "bfloat16")
    flops = (w.benchmark or {}).get("measured_flops") or UNBENCHMARKED_FLOPS.get(w.backend, 5e10)
    return WorkerProfile(w.worker_id, w.name, w.backend, total, float(flops),
                         unified_memory=bool(dev.get("unified_memory")), supported_dtypes=dtypes)


def plan_job(cfg: MeshTrainConfig, workers: list[WorkerRecord], network: NetworkModel | None = None,
             budget_overrides: dict[str, int] | None = None) -> Plan:
    spec = cfg.build_model_spec()
    mb_size = cfg.training.microbatch_size or cfg.training.batch_size
    layers = profile_model(spec, mb_size)
    profiles = [worker_profile(w) for w in workers]
    if cfg.workers.require:
        missing = set(cfg.workers.require) - {p.worker_id for p in profiles} - {p.name for p in profiles}
        if missing:
            raise ValueError(f"required workers not online: {sorted(missing)}")
    pc = cfg.placement
    v1_margin = pc.memory_headroom_fraction is not None or pc.memory_headroom_min_gb is not None
    opts = PlannerOptions(
        optimizer=cfg.training.optimizer,
        num_microbatches=cfg.training.num_microbatches,
        allow_backends=tuple(cfg.workers.allow),
        dtype=cfg.model.dtype,
        headroom_fraction=(pc.memory_headroom_fraction or 0.0) if v1_margin else None,
        headroom_min_bytes=int((pc.memory_headroom_min_gb or 0.0) * 1024**3) if v1_margin else None,
        activation_safety=pc.activation_overhead_factor,
        num_stages=pc.num_stages,
        schedule=cfg.pipeline.schedule,
        max_inflight=cfg.pipeline.max_inflight_microbatches,
        safety_factors=cfg.memory.safety_factors(),
        framework_reserve=cfg.memory.framework_reserve_bytes(),
        budget_overrides=dict(budget_overrides or {}),
    )
    if pc.strategy == "manual":
        return plan_manual(layers, profiles, network or NetworkModel(), opts,
                           [(s.worker, s.layers[0], s.layers[1]) for s in pc.stages])
    return make_plan(pc.strategy, layers, profiles, network, opts)

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
    caps = w.capabilities or {}
    if caps:  # measured by the worker's probes at registration
        dtypes = tuple(dt for dt, key in (("float32", "supports_fp32"), ("float16", "supports_fp16"),
                                          ("bfloat16", "supports_bf16"), ("float64", "supports_fp64"))
                       if caps.get(key))
    else:  # older worker: static assumptions
        dtypes = ("float32", "float16", "bfloat16") if w.backend == "mps" else \
            ("float32", "float16", "bfloat16", "float64")
    bench = w.benchmark or {}
    flops = bench.get("measured_flops") or UNBENCHMARKED_FLOPS.get(w.backend, 5e10)
    # device<->host copy rates (unbenchmarked accelerators: assume ~PCIe 3 x8)
    default_copy = 0.0 if w.backend == "cpu" else 6e9
    return WorkerProfile(w.worker_id, w.name, w.backend, total, float(flops),
                         unified_memory=bool(dev.get("unified_memory")), supported_dtypes=dtypes,
                         supported_ops=caps.get("op_capabilities"),
                         d2h_Bps=float(bench.get("d2h_Bps", default_copy)),
                         h2d_Bps=float(bench.get("h2d_Bps", default_copy)))


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
        async_transport=cfg.transport.async_,
        required_ops=tuple(spec.required_ops) + (("adamw",) if cfg.training.optimizer in ("adam", "adamw") else ()),
    )
    if pc.strategy == "manual":
        return plan_manual(layers, profiles, network or NetworkModel(), opts,
                           [(s.worker, s.layers[0], s.layers[1]) for s in pc.stages])
    return make_plan(pc.strategy, layers, profiles, network, opts)

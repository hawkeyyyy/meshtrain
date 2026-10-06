"""Which layers of a stage stay on the accelerator (HOT) and which live in
local RAM between uses (COLD) -- the V2.4 residency planner.

Device memory model for one stage (see runtime/offload.py, which reserves the
same quantities at startup)::

    HOT layer   : parameters P + gradients P + optimizer state k*P (accelerator optimizer)
    COLD layer  : optimizer state k*P on the device (accelerator optimizer) or nothing (cpu_offload)
    working set : (1 + prefetch_distance) * largest COLD layer   (loaded copies)
                  + largest COLD layer of gradient workspace
                  + optimizer-step workspace (foreach temporaries)
    activations : the V1.5 estimate (saved activations, boundary buffers, workspace)

    fits  <=>  sum <= budget * (1 - fragmentation_margin)     (10%: the CUDA cap applies to the
                                                               caching allocator's *reserved* bytes)

Cost model (per training step, per COLD layer, measured or assumed bandwidth)::

    loads      = 2 * M   (eviction after_use: every microbatch's forward and backward)
               = 1       (after_backward with M = 1, ...); + 1 for the accelerator optimizer step
    T_load     = P / host_to_device_Bps
    T_store    = P / device_to_host_Bps          (writeback after an accelerator optimizer step)
    exposed    = max(0, T_load - overlap_window) with overlap_window = compute time of the
                 previous layer when prefetch_distance >= 1, else 0

Heuristic (documented, not optimal): start with every layer COLD; if that
does not fit, the model cannot train on this budget. Otherwise keep layers HOT
in decreasing order of

    benefit = transfer time saved per step if HOT / extra device bytes if HOT

while everything still fits. ``keep_resident`` (manual) layers are HOT first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meshtrain.planner.graph import LayerProfile, profile_model
from meshtrain.planner.memory import estimate_stage_memory
from meshtrain.runtime.offload import OPTIMIZER_STATE_FACTOR, OPTIMIZER_TEMP_FACTOR, ResidencyPolicy, _normalise_group
from meshtrain.runtime.tensor_store import group_id

GB = 1024**3
DEFAULT_H2D_BPS = {"cuda": 12e9, "mps": 30e9, "cpu": 20e9}   # assumed when not measured
DEFAULT_D2H_BPS = {"cuda": 12e9, "mps": 30e9, "cpu": 20e9}
DEFAULT_FLOPS = {"cuda": 8e12, "mps": 2e12, "cpu": 2e11}


@dataclass
class GroupPlan:
    group_id: str
    layer_index: int
    name: str
    param_bytes: int
    hot: bool
    benefit: float = 0.0
    load_s: float = 0.0
    compute_s: float = 0.0


@dataclass
class ResidencyPlan:
    strategy: str
    budget: int | None
    groups: list[GroupPlan]
    device_bytes: dict[str, int]
    static_required: int
    feasible: bool
    reason: str = ""
    optimizer_execution: str = "accelerator"
    prefetch_distance: int = 1
    estimated_transfer_s: float = 0.0
    estimated_exposed_s: float = 0.0
    estimated_compute_s: float = 0.0
    loads_per_step: int = 0
    ram_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def hot_groups(self) -> tuple[str, ...]:
        return tuple(g.group_id for g in self.groups if g.hot)

    @property
    def device_total(self) -> int:
        return sum(self.device_bytes.values())

    def apply(self, policy: ResidencyPolicy) -> ResidencyPolicy:
        if not self.feasible:
            raise MemoryError(f"no residency plan fits: {self.reason}")
        return ResidencyPolicy(**{**vars(policy), "resident_groups": self.hot_groups})

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "budget": self.budget, "feasible": self.feasible, "reason": self.reason,
                "static_required": self.static_required, "device_bytes": self.device_bytes,
                "device_total": self.device_total, "ram_bytes": self.ram_bytes,
                "optimizer_execution": self.optimizer_execution, "prefetch_distance": self.prefetch_distance,
                "hot": list(self.hot_groups), "cold": [g.group_id for g in self.groups if not g.hot],
                "estimated_transfer_s": self.estimated_transfer_s, "estimated_exposed_s": self.estimated_exposed_s,
                "estimated_compute_s": self.estimated_compute_s, "loads_per_step": self.loads_per_step,
                "notes": self.notes}

    def format(self) -> str:
        gb = fmt_bytes
        lines = [f"Residency plan ({self.strategy}, optimizer {self.optimizer_execution}, "
                 f"prefetch distance {self.prefetch_distance})", ""]
        lines.append(f"  static accelerator requirement   {gb(self.static_required)}")
        lines.append(f"  accelerator budget               {gb(self.budget)}")
        lines.append(f"  planned accelerator use          {gb(self.device_total)}  ({'fits' if self.feasible else 'DOES NOT FIT'})")
        for k, v in self.device_bytes.items():
            if v:
                lines.append(f"      {k:<28} {gb(v)}")
        lines.append(f"  local RAM (offloaded state)      {gb(self.ram_bytes)}")
        hot = [g for g in self.groups if g.hot]
        cold = [g for g in self.groups if not g.hot]
        lines.append(f"  HOT  ({len(hot):>3} layers)  " + _ranges(hot))
        lines.append(f"  COLD ({len(cold):>3} layers)  " + _ranges(cold))
        if cold:
            lines.append(f"  estimated per step: transfers {self.estimated_transfer_s * 1000:.1f} ms "
                         f"({self.loads_per_step} loads per COLD layer), exposed {self.estimated_exposed_s * 1000:.1f} ms, "
                         f"compute {self.estimated_compute_s * 1000:.1f} ms")
        if self.reason:
            lines.append(f"  {self.reason}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "unlimited"
    return f"{n / GB:.2f} GB" if n >= 0.5 * GB else f"{n / 1024**2:.1f} MB"


def _ranges(groups: list[GroupPlan]) -> str:
    if not groups:
        return "-"
    idx = sorted(g.layer_index for g in groups)
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is None or i != prev + 1:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            if i is not None:
                start = i
        if i is not None:
            prev = i
    return "layers " + ", ".join(out)


def plan_residency(layers: list[LayerProfile], *, budget: int | None, policy: ResidencyPolicy, optimizer: str,
                   backend: str = "cuda", num_microbatches: int = 1, schedule: str = "1f1b",
                   stage_index: int = 0, num_stages: int = 1, h2d_Bps: float | None = None,
                   d2h_Bps: float | None = None, flops: float | None = None,
                   fragmentation_margin: float = 0.10, activation_safety: float = 1.25) -> ResidencyPlan:
    """Plan HOT/COLD layers of one stage (``layers`` = that stage's profiles)."""
    sf, tf = OPTIMIZER_STATE_FACTOR[optimizer], OPTIMIZER_TEMP_FACTOR[optimizer]
    cpu_opt = policy.optimizer_execution == "cpu_offload"
    h2d = h2d_Bps or DEFAULT_H2D_BPS.get(backend, 12e9)
    d2h = d2h_Bps or DEFAULT_D2H_BPS.get(backend, 12e9)
    fl = flops or DEFAULT_FLOPS.get(backend, 1e12)
    est = estimate_stage_memory(layers, total=budget or 1 << 50, optimizer=optimizer,
                                num_microbatches=num_microbatches, is_last=stage_index == num_stages - 1,
                                stage_index=stage_index, num_stages=num_stages, schedule=schedule, backend=backend,
                                safety_factor=1.0, framework_reserve=0, activation_safety=activation_safety)
    activations = (est.saved_activations + est.input_buffers + est.output_buffers + est.transport_buffers
                   + est.temporary_workspace)
    static_required = est.required
    with_params = [l for l in layers if l.param_bytes > 0]
    loads = (2 * num_microbatches if policy.eviction == "after_use" else 1) + (0 if cpu_opt else 1)
    groups = []
    for l in with_params:
        compute_s = 3 * l.flops * num_microbatches / fl     # forward + ~2x backward
        groups.append(GroupPlan(group_id(l.index), l.index, l.name, l.param_bytes, hot=False,
                                load_s=l.param_bytes / h2d, compute_s=compute_s))

    def device_bytes(hot: set[str]) -> dict[str, int]:
        hot_g = [g for g in groups if g.group_id in hot]
        cold_g = [g for g in groups if g.group_id not in hot]
        p_hot = sum(g.param_bytes for g in hot_g)
        biggest = max((g.param_bytes for g in cold_g), default=0)
        d = {
            "hot parameters": p_hot,
            "hot gradients": p_hot,
            "hot optimizer state": 0 if cpu_opt else int(sf * p_hot),
            "cold optimizer state": 0 if cpu_opt else int(sf * sum(g.param_bytes for g in cold_g)),
            "loaded cold layers": (1 + policy.prefetch_distance) * biggest,
            "gradient workspace": biggest,
            "optimizer workspace": 0 if cpu_opt else int(tf * max(p_hot, biggest)),
            "activations (V1.5 estimate)": int(activations),
        }
        return d

    usable = None if budget is None else int(budget * (1 - fragmentation_margin))
    fits = (lambda hot: True) if usable is None else (lambda hot: sum(device_bytes(hot).values()) <= usable)
    forced = set()
    if policy.strategy == "manual_offload":
        forced = {_normalise_group(k) for k in policy.keep_resident} & {g.group_id for g in groups}
    elif policy.strategy == "static":
        forced = {g.group_id for g in groups}
    hot = set(forced)
    notes = []
    if policy.strategy == "auto_offload":
        for g in groups:
            saved = loads * g.load_s + (0 if cpu_opt else g.param_bytes / d2h)
            extra = sum(device_bytes(hot | {g.group_id}).values()) - sum(device_bytes(hot).values())
            g.benefit = saved / max(extra, 1)
        for g in sorted(groups, key=lambda g: (-g.benefit, g.layer_index)):
            if g.group_id not in hot and fits(hot | {g.group_id}):
                hot.add(g.group_id)
    for g in groups:
        g.hot = g.group_id in hot
    dev = device_bytes(hot)
    feasible = fits(hot)
    reason = ""
    if not feasible:
        reason = (f"needs {fmt_bytes(sum(dev.values()))} on the accelerator with "
                  f"{'every layer resident' if policy.strategy == 'static' else 'this residency'}, budget "
                  f"{fmt_bytes(budget)} (usable {fmt_bytes(usable)})")
    cold = [g for g in groups if not g.hot]
    transfer = sum(loads * g.load_s + (0 if cpu_opt else g.param_bytes / d2h) for g in cold)
    exposed = 0.0
    for i, g in enumerate(groups):
        if g.hot:
            continue
        window = groups[i - 1].compute_s / num_microbatches if (i > 0 and policy.prefetch_distance > 0) else 0.0
        exposed += loads * max(0.0, g.load_s - window)
    ram = sum(g.param_bytes for g in cold) * 2           # masters + gradient accumulators
    if cpu_opt:
        ram = sum(g.param_bytes for g in groups) * (2 + sf)   # masters, gradients, optimizer state
    if cold and policy.eviction == "after_use" and num_microbatches > 1:
        notes.append(f"after_use eviction reloads each offloaded layer {2 * num_microbatches} times per step "
                     f"({num_microbatches} microbatches): fewer microbatches = less transfer")
    return ResidencyPlan(policy.strategy, budget, groups, dev, static_required, feasible, reason,
                         policy.optimizer_execution, policy.prefetch_distance, transfer, exposed,
                         sum(g.compute_s for g in groups), loads if cold else 0, ram, notes)


def plan_stage_residency(spec, start: int, end: int, *, microbatch_size: int, budget: int | None,
                         policy: ResidencyPolicy, optimizer: str, backend: str = "cuda", num_microbatches: int = 1,
                         schedule: str = "1f1b", stage_index: int = 0, num_stages: int = 1,
                         h2d_Bps: float | None = None, d2h_Bps: float | None = None,
                         flops: float | None = None) -> ResidencyPlan:
    layers = profile_model(spec, microbatch_size)[start:end]
    return plan_residency(layers, budget=budget, policy=policy, optimizer=optimizer, backend=backend,
                          num_microbatches=num_microbatches, schedule=schedule, stage_index=stage_index,
                          num_stages=num_stages, h2d_Bps=h2d_Bps, d2h_Bps=d2h_Bps, flops=flops)


def resolve_stage_policy(policy: ResidencyPolicy | None, spec, start: int, end: int, *, microbatch_size: int,
                         budget: int | None, optimizer: str, backend: str, num_microbatches: int = 1,
                         schedule: str = "1f1b", stage_index: int = 0, num_stages: int = 1,
                         h2d_Bps: float | None = None, d2h_Bps: float | None = None,
                         flops: float | None = None) -> tuple[ResidencyPolicy | None, ResidencyPlan | None]:
    """Fill in ``resident_groups`` for auto_offload; other strategies pass through."""
    if policy is None or policy.strategy != "auto_offload" or policy.resident_groups is not None:
        return policy, None
    plan = plan_stage_residency(spec, start, end, microbatch_size=microbatch_size, budget=budget, policy=policy,
                                optimizer=optimizer, backend=backend, num_microbatches=num_microbatches,
                                schedule=schedule, stage_index=stage_index, num_stages=num_stages,
                                h2d_Bps=h2d_Bps, d2h_Bps=d2h_Bps, flops=flops)
    return plan.apply(policy), plan


def planned_tensor_records(spec, start: int, end: int, plan: ResidencyPlan | None, *, optimizer: str,
                           owner: str = "planned", stage_index: int = 0):
    """TensorMeta records (no memory allocated: built on the meta device) for a stage under ``plan``.

    ``plan=None`` = static V1.5: everything on the accelerator.
    """
    import torch

    from meshtrain.runtime.tensor_store import MemoryTier, TensorMeta, TensorRole, parameter_id

    acc, ram = MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM
    hot = set(plan.hot_groups) if plan is not None else None
    cpu_opt = plan is not None and plan.optimizer_execution == "cpu_offload"
    sf = OPTIMIZER_STATE_FACTOR[optimizer]
    records = []
    for i in range(start, end):
        with torch.device("meta"):
            layer = spec._make_layer(i).to(spec.dtype)
        gid = group_id(i)
        is_hot = hot is None or gid in hot
        for n, p in layer.named_parameters():
            tid = parameter_id(i, n)
            nbytes = p.numel() * p.element_size()
            dtype = str(p.dtype).replace("torch.", "")
            ptier = acc if is_hot else ram
            pcached = {acc} if (is_hot and cpu_opt) else set()
            records.append(TensorMeta(tid, owner, tuple(p.shape), dtype, nbytes, TensorRole.PARAMETER,
                                      ram if pcached else ptier, stage_index, gid, pcached))
            # HOT gradients accumulate on the device (cpu_offload copies them to RAM for the CPU step).
            records.append(TensorMeta(tid + ".grad", owner, tuple(p.shape), dtype, nbytes, TensorRole.GRADIENT,
                                      acc if is_hot else ram, stage_index, gid, {ram} if (is_hot and cpu_opt) else set()))
            if sf:
                stier = ram if cpu_opt else acc
                for key in ("exp_avg", "exp_avg_sq"):
                    records.append(TensorMeta(f"{tid}.optim.{key}", owner, tuple(p.shape), dtype, nbytes,
                                              TensorRole.OPTIMIZER_STATE, stier, stage_index, gid))
    return records

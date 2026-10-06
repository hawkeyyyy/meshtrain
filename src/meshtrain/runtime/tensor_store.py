"""Tensor identity, residency and the TensorStore (V2).

V1.5 introduced the vocabulary (stable tensor ids, ``MemoryTier``,
``TensorRole``) without moving anything. V2 separates *where a tensor's
state lives* from *where its layer computes*:

    compute_owner(layer)  = the stage's worker (unchanged from V1.5)
    residency(tensor)     = a memory tier on that worker (V2)

Every tracked tensor has a ``TensorMeta`` record:

* ``tensor_id``      stable global name, e.g. ``model.layers.17.qkv.weight``,
                     ``model.layers.17.qkv.weight.grad``,
                     ``model.layers.17.qkv.weight.optim.exp_avg``,
                     ``act.s3.mb2.stage1.input``
* ``tier``           the *authoritative* location (exactly one per tensor)
* ``cached``         other tiers holding a valid copy
* ``dirty``          a cached accelerator copy is newer than the authoritative
                     RAM copy and must be written back before it is dropped
* ``version``        incremented on every update of the logical value
* ``pinned``         may not be evicted
* ``group``          offload unit (one layer, see runtime/offload.py)
* role, shape, dtype, bytes, owner worker, compute stage, last/next access

Implemented tiers: ``LOCAL_ACCELERATOR`` and ``LOCAL_RAM``. The other tiers
exist in the enum so interfaces are stable, but every operation that would
place a tensor there raises ``NotImplementedError`` -- they are not faked.

Byte accounting is incremental and per tier/role; ``stats()`` reports
resident bytes, peaks, bytes in flight, pinned host bytes and the configured
accelerator budget. ``reserve_accelerator`` enforces the budget for model
state (parameters, gradients, optimizer state, temporaries) *including*
transfer staging: a copy that briefly needs old + new storage must reserve
both. Activations are tracked and reported but not budget-enforced here (the
planner reserves an activation allowance; see docs/v2-architecture.md).

For a CPU-backend stage the compute memory *is* host RAM: static stages keep
reporting ``LOCAL_RAM`` (V1.5 behaviour). An offloading CPU stage treats its
working set as ``LOCAL_ACCELERATOR`` logically, so the offload machinery runs
(with real copies) and can be tested without a GPU.
"""

from __future__ import annotations

import abc
import enum
import threading
import time
from dataclasses import dataclass, field

import torch


class MemoryTier(enum.Enum):
    LOCAL_ACCELERATOR = "local_accelerator"
    LOCAL_RAM = "local_ram"
    REMOTE_ACCELERATOR = "remote_accelerator"
    REMOTE_RAM = "remote_ram"
    LOCAL_NVME = "local_nvme"
    REMOTE_NVME = "remote_nvme"
    RECOMPUTE = "recompute"                   # dropped, recomputed from an earlier checkpoint


IMPLEMENTED_TIERS = frozenset({MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM})


def require_implemented(tier: MemoryTier) -> MemoryTier:
    if tier not in IMPLEMENTED_TIERS:
        raise NotImplementedError(f"memory tier {tier.value} is not implemented "
                                  f"(V2 local-memory phase supports local_accelerator and local_ram only)")
    return tier


class TensorRole(enum.Enum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATION = "activation"
    ACTIVATION_GRADIENT = "activation_gradient"
    TEMPORARY = "temporary"


# Roles whose accelerator bytes count against ``accelerator_budget``.
BUDGETED_ROLES = (TensorRole.PARAMETER, TensorRole.GRADIENT, TensorRole.OPTIMIZER_STATE, TensorRole.TEMPORARY)

ROLE_ABBREV = {TensorRole.PARAMETER: "PARAM", TensorRole.GRADIENT: "GRAD", TensorRole.OPTIMIZER_STATE: "OPTIM",
               TensorRole.ACTIVATION: "ACT", TensorRole.ACTIVATION_GRADIENT: "ACT_GRAD",
               TensorRole.TEMPORARY: "TEMP"}


class MemoryBudgetError(MemoryError):
    """A transfer would exceed the configured accelerator budget."""

    def __init__(self, tensor_id: str, source: MemoryTier | None, destination: MemoryTier, requested: int,
                 available: int, budget: int):
        self.tensor_id, self.source, self.destination = tensor_id, source, destination
        self.requested, self.available, self.budget = requested, available, budget
        src = source.value if source else "new allocation"
        super().__init__(f"cannot place {tensor_id} ({src} -> {destination.value}): requested "
                         f"{requested / 1024**2:.1f} MB, available {available / 1024**2:.1f} MB "
                         f"of a {budget / 1024**2:.1f} MB accelerator budget")


@dataclass
class TensorMeta:
    tensor_id: str
    owner: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    role: TensorRole
    tier: MemoryTier                       # authoritative location
    compute_stage: int | None = None
    group: str | None = None               # offload unit (layer), e.g. "model.layers.12"
    cached: set = field(default_factory=set)  # tiers with a valid non-authoritative copy
    dirty: bool = False
    pinned: bool = False
    version: int = 0
    last_access: float | None = None
    next_expected_access: int | None = None

    # spec vocabulary
    @property
    def size_bytes(self) -> int:
        return self.bytes

    @property
    def owner_worker(self) -> str:
        return self.owner

    @property
    def current_location(self) -> MemoryTier:
        return self.tier

    def resident_tiers(self) -> set:
        return {self.tier} | set(self.cached)

    def to_dict(self) -> dict:
        return {"tensor_id": self.tensor_id, "owner": self.owner, "shape": list(self.shape), "dtype": self.dtype,
                "bytes": self.bytes, "role": self.role.value, "tier": self.tier.value,
                "cached": sorted(t.value for t in self.cached), "dirty": self.dirty, "pinned": self.pinned,
                "version": self.version, "group": self.group, "compute_stage": self.compute_stage,
                "last_access": self.last_access, "next_expected_access": self.next_expected_access}


def parameter_id(layer_index: int, param_name: str) -> str:
    """Global, stable name of a parameter: ``model.layers.<global layer>.<name inside the layer>``."""
    return f"model.layers.{layer_index}.{param_name}"


def group_id(layer_index: int) -> str:
    return f"model.layers.{layer_index}"


def activation_id(step: int, microbatch: int, stage: int, kind: str = "input") -> str:
    return f"act.s{step}.mb{microbatch}.stage{stage}.{kind}"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class TensorStore(abc.ABC):
    """Where named tensors live and how they move between memory tiers."""

    @abc.abstractmethod
    def register(self, tensor_id: str, tensor: torch.Tensor, role: TensorRole, tier: MemoryTier | None = None,
                 **meta) -> TensorMeta: ...

    @abc.abstractmethod
    def put(self, tensor_id: str, tensor: torch.Tensor, role: TensorRole, tier: MemoryTier | None = None) -> TensorMeta: ...

    @abc.abstractmethod
    def get(self, tensor_id: str) -> torch.Tensor: ...

    @abc.abstractmethod
    def locate(self, tensor_id: str) -> TensorMeta: ...

    @abc.abstractmethod
    def move(self, tensor_id: str, to: MemoryTier) -> TensorMeta: ...

    @abc.abstractmethod
    def prefetch(self, tensor_id: str) -> None: ...

    @abc.abstractmethod
    def evict(self, tensor_id: str, to: MemoryTier) -> None: ...

    @abc.abstractmethod
    def pin(self, tensor_id: str) -> None: ...

    @abc.abstractmethod
    def unpin(self, tensor_id: str) -> None: ...

    @abc.abstractmethod
    def stats(self) -> dict: ...


class LocalTensorStore(TensorStore):
    """Registry + byte accounting for the tensors of one local pipeline stage.

    Standalone tensors (``put``) are held by the store and ``move`` copies them
    between the two local tiers. Tensors that belong to an offload *group*
    (layer parameters) are moved by the stage's ``ResidencyManager``; the
    store's ``move``/``prefetch``/``evict`` delegate to it, so a request for one
    parameter moves its whole layer (layer-level offload).
    """

    def __init__(self, owner: str, device_type: str = "cpu", *, device=None, compute_stage: int | None = None,
                 accelerator_budget: int | None = None, offload: bool = False):
        self.owner = owner
        self.device_type = device_type
        self.device = device                       # DeviceAdapter used for standalone moves (optional)
        self.compute_stage = compute_stage
        # Where computation reads tensors from. A static CPU stage computes from RAM (V1.5).
        self.home = MemoryTier.LOCAL_RAM if device_type == "cpu" and not offload else MemoryTier.LOCAL_ACCELERATOR
        self.accelerator_budget = accelerator_budget
        self.manager = None                        # ResidencyManager when offloading
        self._tensors: dict[str, torch.Tensor] = {}
        self._meta: dict[str, TensorMeta] = {}
        self._lock = threading.RLock()
        self._tier_bytes: dict[MemoryTier, int] = {t: 0 for t in IMPLEMENTED_TIERS}
        self._role_tier_bytes: dict[tuple[TensorRole, MemoryTier], int] = {}
        self._peak: dict[MemoryTier, int] = {t: 0 for t in IMPLEMENTED_TIERS}
        self.in_flight_bytes = 0
        self.in_flight_peak = 0
        self.pinned_host_bytes = 0
        self.budget_violations = 0

    # -- accounting -------------------------------------------------------
    def _account(self, meta: TensorMeta, sign: int) -> None:
        for tier in meta.resident_tiers():
            self._tier_bytes[tier] = self._tier_bytes.get(tier, 0) + sign * meta.bytes
            key = (meta.role, tier)
            self._role_tier_bytes[key] = self._role_tier_bytes.get(key, 0) + sign * meta.bytes
        if sign > 0:
            self._update_peaks()

    def _update_peaks(self) -> None:
        acc = self.accelerator_resident_bytes() + self.in_flight_bytes
        self._peak[MemoryTier.LOCAL_ACCELERATOR] = max(self._peak[MemoryTier.LOCAL_ACCELERATOR], acc)
        self._peak[MemoryTier.LOCAL_RAM] = max(self._peak[MemoryTier.LOCAL_RAM],
                                               self._tier_bytes.get(MemoryTier.LOCAL_RAM, 0))

    def accelerator_resident_bytes(self, budgeted_only: bool = True) -> int:
        tier = MemoryTier.LOCAL_ACCELERATOR
        if not budgeted_only:
            return self._tier_bytes.get(tier, 0)
        return sum(self._role_tier_bytes.get((r, tier), 0) for r in BUDGETED_ROLES)

    def reserve_accelerator(self, nbytes: int, tensor_id: str, source: MemoryTier | None) -> None:
        """Check that ``nbytes`` more model state fits the accelerator budget."""
        if self.accelerator_budget is None:
            return
        used = self.accelerator_resident_bytes() + self.in_flight_bytes
        if used + nbytes > self.accelerator_budget:
            self.budget_violations += 1
            raise MemoryBudgetError(tensor_id, source, MemoryTier.LOCAL_ACCELERATOR, nbytes,
                                    max(0, self.accelerator_budget - used), self.accelerator_budget)

    def begin_transfer(self, nbytes: int) -> None:
        with self._lock:
            self.in_flight_bytes += nbytes
            self.in_flight_peak = max(self.in_flight_peak, self.in_flight_bytes)
            self._update_peaks()

    def end_transfer(self, nbytes: int) -> None:
        with self._lock:
            self.in_flight_bytes = max(0, self.in_flight_bytes - nbytes)

    def reset_peaks(self) -> None:
        with self._lock:
            self._peak = {t: self._tier_bytes.get(t, 0) for t in IMPLEMENTED_TIERS}
            self._peak[MemoryTier.LOCAL_ACCELERATOR] = self.accelerator_resident_bytes() + self.in_flight_bytes
            self.in_flight_peak = self.in_flight_bytes

    # -- registration ---------------------------------------------------------
    def register(self, tensor_id, tensor, role, tier=None, **meta):
        """Track ``tensor`` (the store keeps a reference). Re-registering replaces the record."""
        tier = require_implemented(tier or self.home)
        record = TensorMeta(tensor_id, self.owner, tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""),
                            tensor_nbytes(tensor), role, tier, compute_stage=self.compute_stage, **meta)
        with self._lock:
            old = self._meta.get(tensor_id)
            if old is not None:
                self._account(old, -1)
                record.version = old.version + 1
            self._tensors[tensor_id] = tensor
            self._meta[tensor_id] = record
            self._account(record, +1)
        return record

    def put(self, tensor_id, tensor, role, tier=None):
        return self.register(tensor_id, tensor, role, tier)

    def discard(self, tensor_id: str) -> None:
        with self._lock:
            self._tensors.pop(tensor_id, None)
            meta = self._meta.pop(tensor_id, None)
            if meta is not None:
                self._account(meta, -1)

    # -- lookup -----------------------------------------------------------------
    def get(self, tensor_id):
        try:
            t = self._tensors[tensor_id]
        except KeyError:
            raise KeyError(f"tensor {tensor_id!r} is not owned by {self.owner}") from None
        self._meta[tensor_id].last_access = time.time()
        return t

    def locate(self, tensor_id):
        try:
            return self._meta[tensor_id]
        except KeyError:
            raise KeyError(f"tensor {tensor_id!r} is not owned by {self.owner}") from None

    def __contains__(self, tensor_id: str) -> bool:
        return tensor_id in self._meta

    def ids(self, role: TensorRole | None = None) -> list[str]:
        with self._lock:
            return sorted(k for k, m in self._meta.items() if role is None or m.role == role)

    def records(self, role: TensorRole | None = None) -> list[TensorMeta]:
        with self._lock:
            return [self._meta[k] for k in sorted(self._meta) if role is None or self._meta[k].role == role]

    # -- residency transitions (called by the ResidencyManager) -------------------
    def set_residency(self, tensor_id: str, tier: MemoryTier, cached=(), *, dirty: bool | None = None,
                      bump_version: bool = False, tensor: torch.Tensor | None = None) -> TensorMeta:
        """Record a completed transition. The caller has already moved the bytes."""
        require_implemented(tier)
        for t in cached:
            require_implemented(t)
        with self._lock:
            meta = self.locate(tensor_id)
            if meta.dirty and dirty is None and tier != MemoryTier.LOCAL_ACCELERATOR \
                    and MemoryTier.LOCAL_ACCELERATOR not in cached:
                raise RuntimeError(f"refusing to drop dirty accelerator copy of {tensor_id} without writeback")
            self._account(meta, -1)
            meta.tier, meta.cached = tier, set(cached) - {tier}
            if dirty is not None:
                meta.dirty = dirty
            if bump_version:
                meta.version += 1
            meta.last_access = time.time()
            if tensor is not None:
                self._tensors[tensor_id] = tensor
            self._account(meta, +1)
            return meta

    def mark_dirty(self, tensor_id: str) -> TensorMeta:
        """The accelerator copy was modified in place: it becomes authoritative and newer than RAM."""
        with self._lock:
            meta = self.locate(tensor_id)
            if MemoryTier.LOCAL_ACCELERATOR not in meta.resident_tiers():
                raise RuntimeError(f"{tensor_id} has no accelerator copy to modify")
            ram_copy = MemoryTier.LOCAL_RAM in meta.resident_tiers()
            self._account(meta, -1)
            meta.tier = MemoryTier.LOCAL_ACCELERATOR
            meta.cached = set()       # the RAM copy (if any) is stale; it stays allocated but invalid
            meta.dirty = ram_copy
            meta.version += 1
            self._account(meta, +1)
            return meta

    # -- TensorStore operations -----------------------------------------------------
    def move(self, tensor_id, to):
        require_implemented(to)
        meta = self.locate(tensor_id)
        if meta.group and self.manager is not None:
            if to == MemoryTier.LOCAL_ACCELERATOR:
                self.manager.load(meta.group, reason="store.move")
            else:
                self.manager.unload(meta.group, reason="store.move")
            return self.locate(tensor_id)
        if meta.tier == to and not meta.cached:
            return meta
        if meta.pinned and to != meta.tier:
            raise RuntimeError(f"{tensor_id} is pinned in {meta.tier.value}")
        src = self._tensors[tensor_id]
        if to == MemoryTier.LOCAL_ACCELERATOR:
            if self.device_type != "cpu" or self.home == MemoryTier.LOCAL_ACCELERATOR:
                self.reserve_accelerator(meta.bytes, tensor_id, meta.tier)
            dst = self.device.move_tensor(src) if self.device is not None else src.clone()
        else:
            dst = src.detach().to("cpu", copy=True)
        if self.device is not None:
            self.device.synchronize()
        return self.set_residency(tensor_id, to, (), dirty=False, tensor=dst)

    def prefetch(self, tensor_id):
        meta = self.locate(tensor_id)
        if meta.group and self.manager is not None:
            self.manager.prefetch_group(meta.group)
        # standalone tensors: resident where they are (moves are explicit)

    def evict(self, tensor_id, to):
        require_implemented(to)
        meta = self.locate(tensor_id)
        if meta.pinned:
            raise RuntimeError(f"{tensor_id} is pinned and cannot be evicted")
        if meta.group and self.manager is not None:
            if to != MemoryTier.LOCAL_RAM:
                raise NotImplementedError("parameter groups can only be evicted to local_ram")
            self.manager.unload(meta.group, reason="store.evict")
            return
        if self.manager is None and meta.role == TensorRole.PARAMETER:
            raise NotImplementedError("static stages do not evict parameters (set memory.strategy to an "
                                      "offload strategy)")
        self.move(tensor_id, to)

    def pin(self, tensor_id):
        with self._lock:
            self.locate(tensor_id).pinned = True

    def unpin(self, tensor_id):
        with self._lock:
            self.locate(tensor_id).pinned = False

    # -- reporting ---------------------------------------------------------------------
    def bytes_by_role(self) -> dict[str, int]:
        out = {r.value: 0 for r in TensorRole}
        with self._lock:
            for m in self._meta.values():
                out[m.role.value] += m.bytes
        return out

    def bytes_by_tier(self) -> dict[str, int]:
        with self._lock:
            return {t.value: self._tier_bytes.get(t, 0) for t in sorted(IMPLEMENTED_TIERS, key=lambda t: t.value)}

    def stats(self) -> dict:
        acc, ram = MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM
        with self._lock:
            def role_bytes(role, tier=None):
                tiers = [tier] if tier else list(IMPLEMENTED_TIERS)
                return sum(self._role_tier_bytes.get((role, t), 0) for t in tiers)

            out = {
                "accelerator_resident": self.accelerator_resident_bytes(),
                "accelerator_resident_all": self._tier_bytes.get(acc, 0),
                "ram_resident": self._tier_bytes.get(ram, 0),
                "in_flight": self.in_flight_bytes,
                "in_flight_peak": self.in_flight_peak,
                "pinned_bytes": self.pinned_host_bytes,
                "parameter_bytes": role_bytes(TensorRole.PARAMETER),
                "gradient_bytes": role_bytes(TensorRole.GRADIENT),
                "optimizer_bytes": role_bytes(TensorRole.OPTIMIZER_STATE),
                "optimizer_accelerator_bytes": role_bytes(TensorRole.OPTIMIZER_STATE, acc),
                "activation_bytes": role_bytes(TensorRole.ACTIVATION),
                "temporary_bytes": role_bytes(TensorRole.TEMPORARY),
                "accelerator_peak": self._peak[acc],
                "ram_peak": self._peak[ram],
                "requested_budget": self.accelerator_budget,
                "headroom": (self.accelerator_budget - self._peak[acc]) if self.accelerator_budget else None,
                "budget_violations": self.budget_violations,
                "tensors": len(self._meta),
                "dirty_tensors": sum(1 for m in self._meta.values() if m.dirty),
            }
        return out

    def manifest(self) -> list[dict]:
        with self._lock:
            return [self._meta[k].to_dict() for k in sorted(self._meta)]


def format_tensor_table(records: list[TensorMeta], *, limit: int | None = None) -> str:
    """``meshtrain tensors list`` table plus per-tier totals."""

    def size(n: int) -> str:
        return f"{n / 1024**2:.1f} MB" if n >= 1024**2 else f"{n / 1024:.1f} KB"

    short = {MemoryTier.LOCAL_ACCELERATOR: "ACC", MemoryTier.LOCAL_RAM: "RAM"}
    lines = [f"{'Tensor':<46}{'Role':<10}{'Location':<26}{'Size':>10}", "-" * 92]
    shown = records if limit is None else records[:limit]
    for m in shown:
        loc = m.tier.value.upper() + ("*" if m.dirty else "")
        if m.cached:
            loc += " +" + "+".join(short.get(t, t.value) for t in sorted(m.cached, key=lambda t: t.value))
        lines.append(f"{m.tensor_id[:45]:<46}{ROLE_ABBREV[m.role]:<10}{loc:<26}{size(m.bytes):>10}")
    if limit is not None and len(records) > limit:
        lines.append(f"... {len(records) - limit} more")
    totals: dict[MemoryTier, int] = {}
    for m in records:
        for t in m.resident_tiers():
            totals[t] = totals.get(t, 0) + m.bytes
    gb = lambda n: f"{n / 1024**3:.2f} GB"  # noqa: E731
    lines += ["", "Totals:", f"  accelerator     {gb(totals.get(MemoryTier.LOCAL_ACCELERATOR, 0))}",
              f"  RAM             {gb(totals.get(MemoryTier.LOCAL_RAM, 0))}"]
    if any(m.cached for m in records):
        lines.append("  +ACC / +RAM: a cached copy also exists there (Location = authoritative copy)")
    if any(m.dirty for m in records):
        lines.append("  * dirty: accelerator copy newer than RAM")
    return "\n".join(lines)

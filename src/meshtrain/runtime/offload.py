"""Tensor residency inside one pipeline stage (V2): local-RAM parameter
offload, prefetch, eviction, writeback and optimizer offload.

Compute ownership is unchanged from V1.5: every layer of a stage computes on
that stage's device for the whole job. What V2 changes is where a layer's
*state* lives between uses.

Unit of movement
----------------
A ``ParameterGroup`` is one layer of the stage (``model.layers.<global index>``).
A group is either

* HOT  -- resident on the device for the whole job (V1.5 behaviour), or
* COLD -- its authoritative parameters live in host RAM (pinned on CUDA) and a
  device working copy exists only while the layer is needed.

Parameter identity (the design choice, see docs/v2-architecture.md)
---------------------------------------------------------------------
``nn.Parameter`` objects are never replaced: optimizers, autograd and the
module all keep referring to the same objects. Only ``param.data`` is
re-pointed between the host master tensor and a device working copy. Values
are copied, never aliased across tiers, so exactly one copy is authoritative:

    COLD, between uses      p.data = host master             (RAM authoritative)
    COLD, loaded            p.data = device copy (clean)      (RAM authoritative, ACCEL cached)
    COLD, after a device    p.data = device copy (dirty)      (ACCEL authoritative, RAM stale)
    optimizer step          -> writeback before the copy may be dropped
    HOT, accelerator opt    p.data = device copy              (ACCEL authoritative, no RAM copy)
    HOT, cpu_offload opt    p.data = device copy (clean)      (RAM master authoritative)

Forward / backward
------------------
* forward pre-hook: make sure the layer is on the device (wait for a prefetch
  or load synchronously), then prefetch the next ``prefetch_distance``
  offloaded layers.
* autograd saved tensors: while a stage runs forward, a ``saved_tensors_hooks``
  pack replaces every saved COLD parameter (or view of one) by a reference.
  Unpacking it during backward reloads the layer on demand. The graph therefore
  never pins device copies, so a layer can be evicted right after its forward
  (``eviction: after_use``) and still be differentiated correctly.
* gradients: a post-accumulate-grad hook notices when every parameter of a
  COLD layer has its gradient for the current microbatch, adds the gradients to
  host-RAM accumulators, frees the device gradients and (policy permitting)
  evicts the layer.

Optimizer
---------
* ``accelerator`` (V1.5): HOT parameters step on the device as before. Each
  COLD layer is loaded, its accumulated gradients copied up, stepped on the
  device (copy becomes DIRTY), written back to RAM and evicted. Optimizer state
  of COLD layers stays on the device.
* ``cpu_offload`` (V2.3): every layer has a host-RAM master copy, gradients and
  optimizer state live in RAM and the update runs on the CPU. HOT layers then
  refresh their device copy from the updated master.

Lifecycle of a COLD group (explicit, validated transitions)::

    RESIDENT_RAM -> PREFETCHING -> RESIDENT_ACCELERATOR -> IN_USE_FORWARD
      -> SAVED_FOR_BACKWARD -> IN_USE_BACKWARD -> (DIRTY_ACCELERATOR -> WRITEBACK)
      -> RESIDENT_ACCELERATOR -> RESIDENT_RAM

Budget
------
The stage's ``LocalTensorStore`` enforces ``accelerator_budget`` for model
state. At startup the manager reserves what will exist on the device for the
whole job (HOT parameters, their gradients and optimizer state, device-side
optimizer state of COLD layers, one layer of gradient workspace and the
optimizer-step workspace); each load reserves its parameter bytes while in
flight and while resident. A load that does not fit first evicts idle layers
(least recently used) and otherwise fails with ``MemoryBudgetError`` naming the
tensor, source and destination tier, requested and available bytes.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from meshtrain.runtime.tensor_store import (
    LocalTensorStore,
    MemoryBudgetError,
    MemoryTier,
    TensorRole,
    group_id,
    parameter_id,
    tensor_nbytes,
)

ACC, RAM = MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM
OPTIMIZER_STATE_FACTOR = {"sgd": 0.0, "adam": 2.0, "adamw": 2.0}
OPTIMIZER_TEMP_FACTOR = {"sgd": 0.0, "adam": 1.0, "adamw": 1.0}


class GroupState(enum.Enum):
    RESIDENT_RAM = "resident_ram"
    PREFETCHING = "prefetching"
    RESIDENT_ACCELERATOR = "resident_accelerator"
    IN_USE_FORWARD = "in_use_forward"
    SAVED_FOR_BACKWARD = "saved_for_backward"
    IN_USE_BACKWARD = "in_use_backward"
    DIRTY_ACCELERATOR = "dirty_accelerator"
    WRITEBACK = "writeback"


_S = GroupState
_TRANSITIONS: dict[GroupState, set[GroupState]] = {
    _S.RESIDENT_RAM: {_S.PREFETCHING, _S.RESIDENT_ACCELERATOR},
    _S.PREFETCHING: {_S.RESIDENT_ACCELERATOR},
    _S.RESIDENT_ACCELERATOR: {_S.IN_USE_FORWARD, _S.IN_USE_BACKWARD, _S.DIRTY_ACCELERATOR, _S.RESIDENT_RAM},
    _S.IN_USE_FORWARD: {_S.SAVED_FOR_BACKWARD, _S.RESIDENT_ACCELERATOR},
    _S.SAVED_FOR_BACKWARD: {_S.IN_USE_FORWARD, _S.IN_USE_BACKWARD, _S.RESIDENT_ACCELERATOR, _S.RESIDENT_RAM},
    _S.IN_USE_BACKWARD: {_S.SAVED_FOR_BACKWARD, _S.RESIDENT_ACCELERATOR},
    _S.DIRTY_ACCELERATOR: {_S.WRITEBACK},
    _S.WRITEBACK: {_S.RESIDENT_ACCELERATOR},
}
_LOADED = {_S.RESIDENT_ACCELERATOR, _S.IN_USE_FORWARD, _S.SAVED_FOR_BACKWARD, _S.IN_USE_BACKWARD,
           _S.DIRTY_ACCELERATOR, _S.WRITEBACK}


@dataclass
class ResidencyPolicy:
    strategy: str = "static"                 # static | manual_offload | auto_offload
    keep_resident: tuple[str, ...] = ()      # manual_offload: layers that stay on the device
    resident_groups: tuple[str, ...] | None = None  # auto_offload: chosen by planner/residency.py
    prefetch_distance: int = 1
    eviction: str = "after_use"              # after_use | after_backward
    optimizer_execution: str = "accelerator"  # accelerator | cpu_offload
    pin_host_memory: bool = True
    enforce_allocator_limit: bool = True
    budget_mb: float | None = None
    budget_spec: str | None = None           # "auto" | "6GB" | ...

    @property
    def active(self) -> bool:
        """False = exact V1.5 code path."""
        return self.strategy != "static" or self.optimizer_execution == "cpu_offload"

    def budget_bytes(self, device_total: int | None, backend: str) -> int | None:
        from meshtrain.config import MemoryConfig

        return MemoryConfig(accelerator_budget_mb=self.budget_mb, accelerator_budget=self.budget_spec) \
            .budget_bytes(device_total, backend)

    def is_cold(self, layer_index: int) -> bool:
        gid = group_id(layer_index)
        if self.strategy == "static":
            return False
        if self.strategy == "auto_offload":
            if self.resident_groups is None:
                raise ValueError("auto_offload needs a residency plan (planner/residency.py) before the stage starts")
            return gid not in self.resident_groups
        keep = {_normalise_group(k) for k in self.keep_resident}
        return gid not in keep


def _normalise_group(name) -> str:
    s = str(name).strip()
    if s.isdigit():
        return group_id(int(s))
    if s.startswith("layers."):
        s = "model." + s
    return s


class _ParamRef:
    """What autograd stores instead of a COLD parameter (or a view of one)."""

    __slots__ = ("group", "param", "view")

    def __init__(self, group, param, view):
        self.group, self.param, self.view = group, param, view


@dataclass
class ParameterGroup:
    group_id: str
    layer_index: int
    module: nn.Module
    params: list[tuple[str, nn.Parameter]]
    offloaded: bool
    state: GroupState = GroupState.RESIDENT_RAM
    host: dict[int, torch.Tensor] = field(default_factory=dict)       # id(p) -> host master
    host_grad: dict[int, torch.Tensor] = field(default_factory=dict)  # id(p) -> host accumulator
    grad_valid: set = field(default_factory=set)                       # id(p) with a valid accumulator
    accumulated: set = field(default_factory=set)                      # id(p) done in this backward
    pending: object = None                                             # in-flight prefetch handle
    pending_issue: float = 0.0
    pending_backward: int = 0
    last_access: float = 0.0
    loaded_at: float | None = None

    @property
    def param_bytes(self) -> int:
        return sum(tensor_nbytes(p) for _, p in self.params)

    @property
    def trainable(self) -> int:
        return sum(1 for _, p in self.params if p.requires_grad)

    def transition(self, new: GroupState) -> None:
        if new == self.state:
            return
        if new not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"{self.group_id}: invalid residency transition {self.state.value} -> {new.value}")
        self.state = new


class ResidencyManager:
    def __init__(self, module: nn.Sequential, device, store: LocalTensorStore, policy: ResidencyPolicy, *,
                 layer_offset: int, optimizer: str, timeline=None):
        self.module, self.device, self.store, self.policy = module, device, store, policy
        self.optimizer_name = optimizer
        self.timeline = timeline
        self.cpu_opt = policy.optimizer_execution == "cpu_offload"
        self.lock = threading.RLock()
        self.groups: list[ParameterGroup] = []
        self.by_param: dict[int, tuple[ParameterGroup, str, nn.Parameter]] = {}
        self._handles = []
        self._staging: dict[tuple, torch.Tensor] = {}
        self.counters: dict[str, float] = {}
        self.totals: dict[str, float] = {}
        self.current_step: int | None = None
        store.manager = self
        for local, layer in enumerate(module):
            params = [(parameter_id(local + layer_offset, n), p) for n, p in layer.named_parameters()]
            if not params:
                continue
            g = ParameterGroup(group_id(local + layer_offset), local + layer_offset, layer, params,
                               offloaded=policy.is_cold(local + layer_offset))
            self.groups.append(g)
            for tid, p in params:
                self.by_param[id(p)] = (g, tid, p)
        self.cold = [g for g in self.groups if g.offloaded]
        self.hot = [g for g in self.groups if not g.offloaded]
        self._materialise()
        for g in self.cold:
            self._handles.append(g.module.register_forward_pre_hook(self._make_pre_forward(g)))
            self._handles.append(g.module.register_forward_hook(self._make_post_forward(g)))
            for _, p in g.params:
                if p.requires_grad:
                    self._handles.append(p.register_post_accumulate_grad_hook(self._on_grad_accumulated))
        self._reset_counters()

    # ------------------------------------------------------------------ setup
    def _materialise(self) -> None:
        dev, store, pin = self.device, self.store, self.policy.pin_host_memory
        for b in self.module.buffers():
            b.data = dev.move_tensor(b.data)
        # Model state that will exist on the device for the whole job.
        reserve: list[tuple[str, int, TensorRole]] = []
        sf, tf = OPTIMIZER_STATE_FACTOR[self.optimizer_name], OPTIMIZER_TEMP_FACTOR[self.optimizer_name]
        for g in self.hot:
            reserve.append((g.group_id + ".grad", g.param_bytes, TensorRole.GRADIENT))
            if not self.cpu_opt:
                reserve.append((g.group_id + ".optim", int(sf * g.param_bytes), TensorRole.OPTIMIZER_STATE))
        if self.hot and not self.cpu_opt:
            reserve.append(("stage.optimizer_step_workspace", int(tf * sum(g.param_bytes for g in self.hot)),
                            TensorRole.TEMPORARY))
        if self.cold:
            biggest = max(g.param_bytes for g in self.cold)
            reserve.append(("offload.grad_workspace", biggest, TensorRole.TEMPORARY))
            if not self.cpu_opt:
                reserve.append(("offload.optimizer_step_workspace", int(tf * biggest), TensorRole.TEMPORARY))
                for g in self.cold:
                    reserve.append((g.group_id + ".optim", int(sf * g.param_bytes), TensorRole.OPTIMIZER_STATE))
        for tid, nbytes, role in reserve:
            if nbytes:
                store.reserve_accelerator(nbytes, tid, None)
                store.register(tid, torch.empty(nbytes, dtype=torch.uint8, device="meta"), role, ACC)
        for g in self.groups:
            for tid, p in g.params:
                if g.offloaded or self.cpu_opt:
                    g.host[id(p)] = master = dev.host_copy(p.data, pinned=pin)
                    store.pinned_host_bytes += tensor_nbytes(master) if master.is_pinned() else 0
                if g.offloaded:
                    p.data = g.host[id(p)]
                    store.register(tid, p, TensorRole.PARAMETER, RAM, group=g.group_id)
                else:
                    store.reserve_accelerator(tensor_nbytes(p), tid, RAM)
                    p.data = dev.move_tensor(p.data)
                    cached = (RAM,) if self.cpu_opt else ()
                    meta = store.register(tid, p, TensorRole.PARAMETER, RAM if self.cpu_opt else ACC,
                                          group=g.group_id, pinned=True)
                    if cached:
                        store.set_residency(tid, RAM, (ACC,), dirty=False)
                    meta.pinned = True
            g.state = GroupState.RESIDENT_RAM if g.offloaded else GroupState.RESIDENT_ACCELERATOR
        if self.device.backend == "cuda":
            self.device.synchronize()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # --------------------------------------------------------------- counters
    _COUNTERS = ("prefetch_count", "prefetch_hits", "prefetch_late", "prefetch_misses", "prefetch_skipped_budget",
                 "resident_hits", "demand_loads", "eviction_count", "bytes_evicted", "writeback_count",
                 "writeback_bytes", "h2d_bytes", "d2h_bytes", "h2d_s", "d2h_s", "stall_s", "sync_load_s",
                 "optimizer_offload_s", "grad_offload_bytes", "pressure_evictions")

    def _reset_counters(self) -> None:
        for k, v in self.counters.items():
            self.totals[k] = self.totals.get(k, 0) + v
        self.counters = {k: 0 for k in self._COUNTERS}

    def _span(self, category: str, start: float, end: float, g: ParameterGroup | None = None, **args) -> None:
        tl = self.timeline
        if tl is not None:
            if g is not None:
                args.setdefault("group", g.group_id)
                args.setdefault("bytes", g.param_bytes)
            tl.add(category, start, end, self.current_step, None, **args)

    def _now(self) -> float:
        return self.timeline.now() if self.timeline is not None else time.time()

    # ------------------------------------------------------------ data movement
    def _make_room(self, nbytes: int, exclude: ParameterGroup) -> None:
        budget = self.store.accelerator_budget
        if budget is None:
            return
        while self.store.accelerator_resident_bytes() + self.store.in_flight_bytes + nbytes > budget:
            idle = [g for g in self.cold if g is not exclude and g.state in
                    (GroupState.RESIDENT_ACCELERATOR, GroupState.SAVED_FOR_BACKWARD)]
            if not idle:
                return  # reserve_accelerator raises with the numbers
            victim = min(idle, key=lambda g: g.last_access)
            self.unload(victim, reason="memory pressure")
            self.counters["pressure_evictions"] += 1

    def _start_load(self, g: ParameterGroup, *, prefetch: bool) -> bool:
        need = g.param_bytes
        tid0 = g.params[0][0]
        if prefetch:
            try:
                self.store.reserve_accelerator(need, tid0, RAM)
            except MemoryBudgetError:
                self.counters["prefetch_skipped_budget"] += 1
                return False
        else:
            self._make_room(need, g)
            self.store.reserve_accelerator(need, tid0, RAM)
        self.store.begin_transfer(need)
        g.pending_issue = self._now()
        g.pending = self.device.offload_h2d_start([g.host[id(p)] for _, p in g.params])
        g.transition(GroupState.PREFETCHING)
        return True

    def _finish_load(self, g: ParameterGroup, *, demand: bool) -> None:
        handle = g.pending
        t0 = self._now()
        late = not self.device.offload_h2d_ready(handle)
        if late or demand:
            self.device.offload_h2d_wait(handle)
        t1 = self._now()
        tensors = self.device.offload_h2d_finish(handle)
        g.pending = None
        need = g.param_bytes
        self.store.end_transfer(need)
        for (tid, p), t in zip(g.params, tensors):
            p.data = t
            self.store.set_residency(tid, RAM, (ACC,), dirty=False)
        g.transition(GroupState.RESIDENT_ACCELERATOR)
        g.loaded_at = t1
        dur = self.device.offload_h2d_seconds(handle)
        c = self.counters
        c["h2d_bytes"] += need
        c["h2d_s"] += dur if dur is not None else (t1 - g.pending_issue)
        if demand:
            c["demand_loads"] += 1
            c["sync_load_s"] += t1 - t0
            self._span("TENSOR_LOAD", t0, t1, g)
        else:
            self._span("TENSOR_PREFETCH", g.pending_issue,
                       g.pending_issue + dur if dur is not None else t1, g)
            if late:
                c["prefetch_late"] += 1
                c["stall_s"] += t1 - t0
                self._span("PREFETCH_STALL", t0, t1, g)

    def prefetch_group(self, gid: str) -> None:
        with self.lock:
            g = next(x for x in self.groups if x.group_id == gid)
            if g.offloaded and g.state == GroupState.RESIDENT_RAM and self._start_load(g, prefetch=True):
                self.counters["prefetch_count"] += 1

    def _prefetch_ahead(self, g: ParameterGroup, direction: int) -> None:
        d = self.policy.prefetch_distance
        if d <= 0:
            return
        order = self.cold if direction > 0 else self.cold[::-1]
        i = order.index(g)
        for nxt in order[i + 1:i + 1 + d]:
            if nxt.state == GroupState.RESIDENT_RAM and self._start_load(nxt, prefetch=True):
                self.counters["prefetch_count"] += 1

    def ensure_loaded(self, g: ParameterGroup, phase: str) -> None:
        """Make ``g`` usable on the device for ``phase`` (forward | backward | optimizer)."""
        with self.lock:
            if g.state in _LOADED:
                self.counters["resident_hits"] += 1
            elif g.state == GroupState.PREFETCHING:
                self._finish_load(g, demand=False)
                self.counters["prefetch_hits"] += 1
            else:
                if self.policy.prefetch_distance > 0:
                    self.counters["prefetch_misses"] += 1
                self._start_load(g, prefetch=False)
                self._finish_load(g, demand=True)
            g.last_access = self._now()
            if phase == "forward":
                g.transition(GroupState.IN_USE_FORWARD)
                self._prefetch_ahead(g, +1)
            elif phase == "backward":
                if g.state != GroupState.IN_USE_BACKWARD:
                    g.transition(GroupState.IN_USE_BACKWARD)
                    self._prefetch_ahead(g, -1)

    def load(self, gid: str, reason: str = "") -> None:
        g = next(x for x in self.groups if x.group_id == gid)
        if g.offloaded:
            self.ensure_loaded(g, "manual")

    def writeback(self, g: ParameterGroup) -> None:
        if g.state != GroupState.DIRTY_ACCELERATOR:
            return
        g.transition(GroupState.WRITEBACK)
        t0 = self._now()
        self.device.offload_d2h([p.data for _, p in g.params], [g.host[id(p)] for _, p in g.params])
        t1 = self._now()
        for tid, _ in g.params:
            self.store.set_residency(tid, RAM, (ACC,), dirty=False)
        g.transition(GroupState.RESIDENT_ACCELERATOR)
        c = self.counters
        c["writeback_count"] += 1
        c["writeback_bytes"] += g.param_bytes
        c["d2h_bytes"] += g.param_bytes
        c["d2h_s"] += t1 - t0
        self._span("TENSOR_WRITEBACK", t0, t1, g)

    def unload(self, g_or_id, reason: str = "") -> None:
        with self.lock:
            g = g_or_id if isinstance(g_or_id, ParameterGroup) else next(
                x for x in self.groups if x.group_id == g_or_id)
            if not g.offloaded:
                raise RuntimeError(f"{g.group_id} is resident (HOT) and is not evicted")
            if g.state == GroupState.RESIDENT_RAM:
                return
            if g.state == GroupState.PREFETCHING:
                self._finish_load(g, demand=False)
            if g.state in (GroupState.IN_USE_FORWARD, GroupState.IN_USE_BACKWARD):
                raise RuntimeError(f"cannot evict {g.group_id} while it is {g.state.value}")
            if g.state == GroupState.DIRTY_ACCELERATOR:
                self.writeback(g)
            t0 = self._now()
            self.device.before_release()
            for tid, p in g.params:
                if self.store.locate(tid).dirty:
                    raise RuntimeError(f"refusing to drop dirty {tid}")
                p.data = g.host[id(p)]
                self.store.set_residency(tid, RAM, ())
            g.transition(GroupState.RESIDENT_RAM)
            self.counters["eviction_count"] += 1
            self.counters["bytes_evicted"] += g.param_bytes
            t1 = self._now()
            self._span("TENSOR_EVICT", t0, t1, g, reason=reason)
            if g.loaded_at is not None:
                self._span("TENSOR_ACCELERATOR_RESIDENT", g.loaded_at, t1, g, thread=f"residency {g.group_id}")
                g.loaded_at = None

    # ------------------------------------------------------------------- hooks
    def _make_pre_forward(self, g: ParameterGroup):
        def hook(module, args):
            self.ensure_loaded(g, "forward")
        return hook

    def _make_post_forward(self, g: ParameterGroup):
        def hook(module, args, output):
            with self.lock:
                if torch.is_grad_enabled():
                    g.pending_backward += 1
                    g.transition(GroupState.SAVED_FOR_BACKWARD)
                else:
                    g.transition(GroupState.RESIDENT_ACCELERATOR)
                if self.policy.eviction == "after_use" or g.pending_backward == 0:
                    self.unload(g, reason="after forward")
        return hook

    def pack(self, t: torch.Tensor):
        """saved_tensors_hooks pack: COLD parameters are saved by reference."""
        entry = self.by_param.get(id(t))
        view = None
        if entry is None and t._base is not None:
            entry = self.by_param.get(id(t._base))
            view = (tuple(t.size()), tuple(t.stride()), t.storage_offset())
        if entry is None or not entry[0].offloaded:
            return None
        return _ParamRef(entry[0], entry[2], view)

    def unpack(self, ref: _ParamRef) -> torch.Tensor:
        self.ensure_loaded(ref.group, "backward")
        base = ref.param.detach()
        return base if ref.view is None else base.as_strided(*ref.view)

    def _on_grad_accumulated(self, p: torch.Tensor) -> None:
        g = self.by_param[id(p)][0]
        with self.lock:
            g.accumulated.add(id(p))
            if len(g.accumulated) >= g.trainable:
                self._group_backward_done(g)

    def _accumulate_host_grad(self, g: ParameterGroup, tid: str, p: nn.Parameter, grad: torch.Tensor) -> None:
        key = id(p)
        host = g.host_grad.get(key)
        t0 = self._now()
        if host is None:
            g.host_grad[key] = host = self.device.host_copy(grad, pinned=self.policy.pin_host_memory)
            if host.is_pinned():
                self.store.pinned_host_bytes += tensor_nbytes(host)
            self.store.register(tid + ".grad", host, TensorRole.GRADIENT, RAM, group=g.group_id)
        elif key not in g.grad_valid:
            self.device.offload_d2h([grad], [host])
        elif grad.device.type == "cpu":
            host.add_(grad)
        else:
            stage = self._staging_buffer(grad)
            self.device.offload_d2h([grad], [stage])
            host.add_(stage)
        g.grad_valid.add(key)
        self.counters["d2h_bytes"] += tensor_nbytes(grad)
        self.counters["grad_offload_bytes"] += tensor_nbytes(grad)
        self.counters["d2h_s"] += self._now() - t0

    def _staging_buffer(self, like: torch.Tensor) -> torch.Tensor:
        key = (tuple(like.shape), like.dtype)
        buf = self._staging.get(key)
        if buf is None:
            buf = self._staging[key] = self.device.host_copy(like, pinned=self.policy.pin_host_memory)
        return buf

    def _group_backward_done(self, g: ParameterGroup) -> None:
        t0 = self._now()
        for tid, p in g.params:
            if p.grad is not None:
                self._accumulate_host_grad(g, tid, p, p.grad)
                p.grad = None
        self._span("OPTIMIZER_OFFLOAD", t0, self._now(), g, what="gradients to RAM")
        g.accumulated.clear()
        g.pending_backward = max(0, g.pending_backward - 1)
        if g.state == GroupState.IN_USE_BACKWARD:
            g.transition(GroupState.SAVED_FOR_BACKWARD if g.pending_backward else GroupState.RESIDENT_ACCELERATOR)
        if g.state in _LOADED and (self.policy.eviction == "after_use" or g.pending_backward == 0):
            self.unload(g, reason="after backward")

    def after_backward_call(self) -> None:
        """Called after each stage backward (one microbatch): finish partially-updated layers."""
        with self.lock:
            for g in self.cold:
                if g.accumulated:
                    self._group_backward_done(g)
                elif g.state == GroupState.IN_USE_BACKWARD:
                    g.transition(GroupState.SAVED_FOR_BACKWARD if g.pending_backward
                                 else GroupState.RESIDENT_ACCELERATOR)

    # --------------------------------------------------------------- optimizer
    def gradient(self, p: nn.Parameter) -> torch.Tensor | None:
        g = self.by_param[id(p)][0]
        if g.offloaded:
            return g.host_grad[id(p)] if id(p) in g.grad_valid else None
        return p.grad

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        with self.lock:
            for g in self.cold:  # nothing may still be referenced by an unfinished microbatch
                g.pending_backward = 0
                if g.state in _LOADED:
                    if g.state in (GroupState.IN_USE_FORWARD, GroupState.IN_USE_BACKWARD,
                                   GroupState.SAVED_FOR_BACKWARD):
                        g.transition(GroupState.RESIDENT_ACCELERATOR)
                    self.unload(g, reason="before optimizer step")
                elif g.state == GroupState.PREFETCHING:
                    self.unload(g, reason="before optimizer step")
            if self.cpu_opt:
                self._cpu_step(optimizer)
            else:
                self._accelerator_step(optimizer)
            self.refresh_records(optimizer)

    def _accelerator_step(self, optimizer) -> None:
        if self.hot:
            optimizer.step()        # COLD parameters have grad None here and are skipped
            for g in self.hot:      # ... and HOT ones must not be stepped again with each COLD layer
                for _, p in g.params:
                    p.grad = None
        for i, g in enumerate(self.cold):
            if not g.grad_valid:
                continue
            self.ensure_loaded(g, "optimizer")
            if self.policy.prefetch_distance > 0:
                for nxt in self.cold[i + 1:i + 1 + self.policy.prefetch_distance]:
                    if nxt.state == GroupState.RESIDENT_RAM and nxt.grad_valid and self._start_load(nxt, prefetch=True):
                        self.counters["prefetch_count"] += 1
            keys = [(tid, p) for tid, p in g.params if id(p) in g.grad_valid]
            t0 = self._now()
            grads = self.device.offload_h2d_finish(self.device.offload_h2d_start([g.host_grad[id(p)] for _, p in keys]))
            self.counters["h2d_bytes"] += sum(tensor_nbytes(t) for t in grads)
            for (_, p), gr in zip(keys, grads):
                p.grad = gr
            self._span("OPTIMIZER_OFFLOAD", t0, self._now(), g, what="gradients to device")
            optimizer.step()        # only this layer's parameters have gradients
            for tid, p in g.params:
                p.grad = None
                self.store.mark_dirty(tid)
            g.transition(GroupState.DIRTY_ACCELERATOR)
            self.writeback(g)
            self.unload(g, reason="after optimizer step")

    def _cpu_step(self, optimizer) -> None:
        t0 = self._now()
        device_copies: dict[int, torch.Tensor] = {}
        for g in self.hot:
            for tid, p in g.params:
                if p.grad is not None:
                    self._accumulate_host_grad(g, tid, p, p.grad)
                    p.grad = None
                device_copies[id(p)] = p.data
                p.data = g.host[id(p)]
        for g in self.groups:
            for _, p in g.params:
                p.grad = g.host_grad[id(p)] if id(p) in g.grad_valid else None
        t1 = self._now()
        optimizer.step()
        t2 = self._now()
        for g in self.groups:
            for tid, p in g.params:
                p.grad = None
        for g in self.hot:
            hosts = [g.host[id(p)] for _, p in g.params]
            for (tid, p), h in zip(g.params, hosts):
                dev_t = device_copies[id(p)]
                dev_t.copy_(h, non_blocking=h.is_pinned())
                p.data = dev_t
                meta = self.store.locate(tid)
                meta.version += 1
            self.counters["h2d_bytes"] += g.param_bytes
        for g in self.cold:
            for tid, _ in g.params:
                self.store.locate(tid).version += 1
        self.device.synchronize()
        t3 = self._now()
        self.counters["optimizer_offload_s"] += t3 - t0
        self._span("OPTIMIZER_OFFLOAD", t0, t1, what="gradients to RAM, swap to masters")
        self._span("OPTIMIZER_OFFLOAD", t2, t3, what="refresh resident layers")

    def zero_grad(self) -> None:
        for g in self.groups:
            g.grad_valid.clear()

    def refresh_records(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Register real optimizer-state / gradient tensors (replacing startup reservations)."""
        store = self.store
        if optimizer is None:
            return
        state_tier = RAM if self.cpu_opt else ACC
        seen_groups = set()
        for g in self.groups:
            for tid, p in g.params:
                for key, v in optimizer.state.get(p, {}).items():
                    if torch.is_tensor(v) and v.dim() > 0:
                        if g.group_id not in seen_groups and g.group_id + ".optim" in store:
                            store.discard(g.group_id + ".optim")
                        store.register(f"{tid}.optim.{key}", v, TensorRole.OPTIMIZER_STATE, state_tier,
                                       group=g.group_id)
                        seen_groups.add(g.group_id)

    # ----------------------------------------------------------------- reports
    def logical_parameter(self, p: nn.Parameter) -> torch.Tensor:
        """Authoritative value of a parameter, wherever it lives."""
        return p.detach()

    def step_begin(self, step: int) -> None:
        self.current_step = step
        self.store.reset_peaks()

    def step_stats(self) -> dict:
        c = dict(self.counters)
        lookups = c["resident_hits"] + c["prefetch_hits"] + c["demand_loads"]
        st = self.store.stats()
        out = {
            "strategy": self.policy.strategy, "optimizer_execution": self.policy.optimizer_execution,
            "prefetch_distance": self.policy.prefetch_distance, "eviction": self.policy.eviction,
            "groups_hot": len(self.hot), "groups_cold": len(self.cold),
            "tensor_prefetch_count": c["prefetch_count"], "tensor_eviction_count": c["eviction_count"],
            "prefetch_hits": c["prefetch_hits"], "prefetch_misses": c["prefetch_misses"],
            "prefetch_late": c["prefetch_late"], "demand_loads": c["demand_loads"],
            "prefetch_skipped_budget": c["prefetch_skipped_budget"], "pressure_evictions": c["pressure_evictions"],
            "bytes_evicted": c["bytes_evicted"], "writeback_bytes": c["writeback_bytes"],
            "writeback_count": c["writeback_count"],
            "H2D_bytes": c["h2d_bytes"], "D2H_bytes": c["d2h_bytes"],
            "H2D_ms": c["h2d_s"] * 1000, "D2H_ms": c["d2h_s"] * 1000,
            "prefetch_stall_ms": c["stall_s"] * 1000, "sync_load_ms": c["sync_load_s"] * 1000,
            "optimizer_offload_ms": c["optimizer_offload_s"] * 1000,
            "tensor_cache_hit_ratio": (c["resident_hits"] + c["prefetch_hits"]) / lookups if lookups else 1.0,
            "accelerator_resident_peak": st["accelerator_peak"], "RAM_resident_peak": st["ram_peak"],
            "accelerator_resident": st["accelerator_resident"], "ram_resident": st["ram_resident"],
            "pinned_bytes": st["pinned_bytes"], "requested_budget": st["requested_budget"],
            "headroom": st["headroom"], "in_flight_peak": st["in_flight_peak"],
            "budget_violations": st["budget_violations"],
        }
        self._reset_counters()
        return out

    def audit(self) -> dict:
        """Recompute residency from the tensors themselves (tests: ledger == reality)."""
        on_device = 0
        problems = []
        for g in self.groups:
            for tid, p in g.params:
                meta = self.store.locate(tid)
                host = g.host.get(id(p))
                on_host_master = host is not None and p.data_ptr() == host.data_ptr()
                dev_resident = p.data.device.type != "cpu" or (
                    self.device.backend == "cpu" and g.offloaded and not on_host_master)
                if g.offloaded:
                    if dev_resident != (ACC in meta.resident_tiers()):
                        problems.append(f"{tid}: ledger {meta.resident_tiers()} vs device={dev_resident}")
                    if not dev_resident and not on_host_master:
                        problems.append(f"{tid}: evicted but not pointing at its host master")
                if dev_resident:
                    on_device += tensor_nbytes(p)
        return {"parameter_bytes_on_device": on_device, "problems": problems}

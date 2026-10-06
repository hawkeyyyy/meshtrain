"""Tensor residency inside one pipeline stage: local-RAM offload (V2) and
remote-RAM backing storage (V2.5).

Compute ownership is unchanged from V1.5: every layer of a stage computes on
that stage's device for the whole job. V2 decides where a layer's *state*
lives between uses; V2.5 adds a third tier, RAM on another machine.

Unit of movement
----------------
A ``ParameterGroup`` is one layer of the stage (``model.layers.<global index>``).
Two independent questions are answered per group:

* device residency -- HOT (on the accelerator for the whole job) or COLD (a
  device working copy exists only while the layer computes);
* home -- where the authoritative master copy lives between uses:
  ``local`` (host RAM of the compute machine) or ``remote`` (REMOTE_RAM on a
  RAM worker, V2.5). Remote homes require the CPU optimizer: masters *and*
  AdamW state live remotely and gradients accumulate in local RAM.

Parameter identity (docs/v2-architecture.md)
--------------------------------------------
``nn.Parameter`` objects are never replaced; only ``param.data`` is re-pointed
between a host tensor, a device working copy or (remote, not staged) an empty
placeholder. Exactly one copy is authoritative at any time.

Lifecycles (explicit, validated)
-------------------------------
Device side (``GroupState``)::

    RESIDENT_RAM -> PREFETCHING -> RESIDENT_ACCELERATOR -> IN_USE_FORWARD
      -> SAVED_FOR_BACKWARD -> IN_USE_BACKWARD -> (DIRTY_ACCELERATOR -> WRITEBACK)
      -> RESIDENT_ACCELERATOR -> RESIDENT_RAM

Host side of remote-homed groups (``HostState``)::

    REMOTE_ONLY -> FETCHING_REMOTE -> LOCAL_STAGED -> (WRITEBACK_REMOTE -> LOCAL_STAGED) -> REMOTE_ONLY

A remote group can be loaded onto the device only from LOCAL_STAGED, and its
staged copy can be dropped only when clean (a dirty staged copy is written
back and its new version committed first).

Forward / backward
------------------
Forward pre-hooks load a COLD layer (wait for its prefetch or load
synchronously) and prefetch ahead -- for remote layers in two stages:
remote RAM -> local staging (``remote_prefetch_distance`` ahead) and local ->
device (``prefetch_distance`` ahead). Saved COLD parameters are packed by
reference and reloaded on demand in backward, so a layer may be evicted right
after its forward. Gradients of COLD layers accumulate in host RAM.

Reuse (V2.5): with ``reuse`` the stage runs layer-major (see
``Stage.layer_major_step``) and calls ``end_window`` once per layer and pass,
so a layer is fetched once for all microbatches of a forward or backward pass
instead of once per microbatch.

Optimizer
---------
* ``accelerator``: HOT parameters step on the device; COLD layers are loaded,
  stepped on the device, written back and evicted.
* ``cpu_offload``: masters, gradients and AdamW state in RAM; layers step one
  at a time on the CPU. Remote-homed layers fetch master + state into local
  staging, step, write params + state back (versioned PUT) and drop staging.

Budgets
-------
``accelerator_budget`` and ``local_ram_budget`` are enforced by the stage's
``LocalTensorStore`` ledger (reservations at startup + every load/stage), the
remote budget by the remote worker's reservation. Failures raise
``MemoryBudgetError`` / ``RemoteMemoryError`` naming tensor, tiers and bytes.
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

ACC, RAM, REMOTE = MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM, MemoryTier.REMOTE_RAM
OPTIMIZER_STATE_FACTOR = {"sgd": 0.0, "adam": 2.0, "adamw": 2.0}
OPTIMIZER_TEMP_FACTOR = {"sgd": 0.0, "adam": 1.0, "adamw": 1.0}
ADAM_KEYS = ("exp_avg", "exp_avg_sq")


class GroupState(enum.Enum):
    RESIDENT_RAM = "resident_ram"
    PREFETCHING = "prefetching"
    RESIDENT_ACCELERATOR = "resident_accelerator"
    IN_USE_FORWARD = "in_use_forward"
    SAVED_FOR_BACKWARD = "saved_for_backward"
    IN_USE_BACKWARD = "in_use_backward"
    DIRTY_ACCELERATOR = "dirty_accelerator"
    WRITEBACK = "writeback"


class HostState(enum.Enum):
    LOCAL = "local"                    # local-homed: the RAM master always exists
    REMOTE_ONLY = "remote_only"
    FETCHING_REMOTE = "fetching_remote"
    LOCAL_STAGED = "local_staged"
    WRITEBACK_REMOTE = "writeback_remote"


_S, _H = GroupState, HostState
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
_HOST_TRANSITIONS: dict[HostState, set[HostState]] = {
    _H.LOCAL: set(),
    _H.REMOTE_ONLY: {_H.FETCHING_REMOTE, _H.LOCAL_STAGED},
    _H.FETCHING_REMOTE: {_H.LOCAL_STAGED},
    _H.LOCAL_STAGED: {_H.REMOTE_ONLY, _H.WRITEBACK_REMOTE, _H.FETCHING_REMOTE},   # fetch AdamW state too
    _H.WRITEBACK_REMOTE: {_H.LOCAL_STAGED},
}
_LOADED = {_S.RESIDENT_ACCELERATOR, _S.IN_USE_FORWARD, _S.SAVED_FOR_BACKWARD, _S.IN_USE_BACKWARD,
           _S.DIRTY_ACCELERATOR, _S.WRITEBACK}


@dataclass
class RemoteSpec:
    """Where REMOTE_RAM lives and how to talk to it (V2.5)."""

    address: str                     # host:port of the RAM worker's tensor service
    worker: str = ""                 # display name (e.g. "fedora")
    token: str | None = None
    job_id: str = "meshtrain-job"
    budget_bytes: int | None = None  # this job's remote RAM budget (reserved up front)
    max_inflight_fetches: int = 2
    max_inflight_writebacks: int = 2
    checksum: str = "none"           # none | crc32
    timeout_s: float = 120.0
    emulate_bandwidth_Bps: float | None = None   # experiments: throttle the link (each direction)
    emulate_latency_s: float = 0.0


@dataclass
class ResidencyPolicy:
    strategy: str = "static"                 # static | manual_offload | auto_offload | remote_offload
    keep_resident: tuple[str, ...] = ()      # manual_offload: layers that stay on the device
    resident_groups: tuple[str, ...] | None = None  # auto/remote: chosen by planner/residency.py
    remote_groups: tuple[str, ...] | None = None    # remote_offload: groups homed in REMOTE_RAM
    prefetch_distance: int = 1
    remote_prefetch_distance: int | None = None     # default prefetch_distance + 1
    eviction: str = "after_use"              # after_use | after_backward
    optimizer_execution: str = "accelerator"  # accelerator | cpu_offload
    pin_host_memory: bool = True
    enforce_allocator_limit: bool = True
    budget_mb: float | None = None
    budget_spec: str | None = None           # "auto" | "6GB" | ...
    local_ram_budget_mb: float | None = None
    reuse: bool = False                      # layer-major execution + keep staged copies while they fit
    remote: RemoteSpec | None = None

    @property
    def active(self) -> bool:
        """False = exact V1.5 code path."""
        return self.strategy != "static" or self.optimizer_execution == "cpu_offload"

    def budget_bytes(self, device_total: int | None, backend: str) -> int | None:
        from meshtrain.config import MemoryConfig

        return MemoryConfig(accelerator_budget_mb=self.budget_mb, accelerator_budget=self.budget_spec) \
            .budget_bytes(device_total, backend)

    @property
    def local_ram_budget(self) -> int | None:
        return int(self.local_ram_budget_mb * 1024**2) if self.local_ram_budget_mb else None

    def is_cold(self, layer_index: int) -> bool:
        gid = group_id(layer_index)
        if self.strategy == "static":
            return False
        if self.strategy in ("auto_offload", "remote_offload"):
            if self.resident_groups is None:
                raise ValueError(f"{self.strategy} needs a residency plan (planner/residency.py) before the "
                                 "stage starts")
            return gid not in self.resident_groups
        keep = {_normalise_group(k) for k in self.keep_resident}
        return gid not in keep

    def is_remote(self, layer_index: int) -> bool:
        return self.strategy == "remote_offload" and group_id(layer_index) in (self.remote_groups or ())


def connect_remote(spec: RemoteSpec):
    """RemoteTensorClient for ``spec`` (optionally behind an emulated slow link)."""
    from meshtrain.networking.tensor_server import RemoteTensorClient

    if spec is None:
        raise ValueError("remote_offload needs memory.remote_ram (a remote RAM worker address)")
    host, _, port = spec.address.rpartition(":")
    wrapper = None
    if spec.emulate_bandwidth_Bps:
        from meshtrain.networking.emulation import EmulatedLink

        bw, lat = spec.emulate_bandwidth_Bps, spec.emulate_latency_s
        wrapper = lambda link: EmulatedLink(link, bw, lat)  # noqa: E731
    client = RemoteTensorClient(host or "127.0.0.1", int(port), token=spec.token, job_id=spec.job_id,
                                worker=spec.worker, max_inflight_fetches=spec.max_inflight_fetches,
                                max_inflight_writebacks=spec.max_inflight_writebacks, checksum=spec.checksum,
                                timeout_s=spec.timeout_s, link_wrapper=wrapper)
    client.start_heartbeat()
    return client


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
    remote: bool = False
    state: GroupState = GroupState.RESIDENT_RAM
    host_state: HostState = HostState.LOCAL
    host: dict[int, torch.Tensor] = field(default_factory=dict)       # id(p) -> host master / staged copy
    host_grad: dict[int, torch.Tensor] = field(default_factory=dict)  # id(p) -> host accumulator
    grad_valid: set = field(default_factory=set)
    accumulated: set = field(default_factory=set)
    pending: object = None                                             # in-flight device prefetch handle
    pending_issue: float = 0.0
    pending_backward: int = 0
    last_access: float = 0.0
    loaded_at: float | None = None
    shapes: dict[int, tuple] = field(default_factory=dict)             # id(p) -> (shape, dtype) (remote)
    versions: dict[str, int] = field(default_factory=dict)             # tensor id -> committed remote version
    fetch: object = None                                               # remote fetch future
    fetch_issue: float = 0.0
    fetch_with_state: bool = False
    staged_state: dict[int, dict] = field(default_factory=dict)        # id(p) -> {key: staged tensor}
    staged_at: float | None = None
    in_window: bool = False

    @property
    def param_bytes(self) -> int:
        return sum(math_bytes(*self.shapes[id(p)]) if id(p) in self.shapes else tensor_nbytes(p)
                   for _, p in self.params)

    @property
    def trainable(self) -> int:
        return sum(1 for _, p in self.params if p.requires_grad)

    def transition(self, new: GroupState) -> None:
        if new == self.state:
            return
        if new not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"{self.group_id}: invalid residency transition {self.state.value} -> {new.value}")
        self.state = new

    def host_transition(self, new: HostState) -> None:
        if new == self.host_state:
            return
        if new not in _HOST_TRANSITIONS[self.host_state]:
            raise RuntimeError(f"{self.group_id}: invalid host transition {self.host_state.value} -> {new.value}")
        self.host_state = new


def math_bytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * torch.empty((), dtype=dtype).element_size()


class ResidencyManager:
    def __init__(self, module: nn.Sequential | None, device, store: LocalTensorStore, policy: ResidencyPolicy, *,
                 layer_offset: int, optimizer: str, timeline=None, layer_factory=None,
                 num_layers: int | None = None, remote_client=None):
        self.device, self.store, self.policy = device, store, policy
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
        self.window_mode = False
        self.client = remote_client
        self._writebacks: list[tuple[ParameterGroup, object, float, dict]] = []
        store.manager = self
        if policy.strategy == "remote_offload":
            if not self.cpu_opt:
                raise ValueError("remote_offload needs optimizer.execution: cpu_offload (masters and AdamW state "
                                 "live remotely)")
            if self.client is None:
                raise ValueError("remote_offload needs a remote tensor client (memory.remote_ram)")
        self._reset_counters()
        self._reserve_static(module, layer_factory, layer_offset, num_layers)
        layers = []
        n = len(module) if module is not None else num_layers
        for local in range(n):
            layer = module[local] if module is not None else layer_factory(local + layer_offset)
            self._adopt(layer, local + layer_offset)
            layers.append(layer)
        self.module = module if module is not None else nn.Sequential(*layers)
        for b in self.module.buffers():
            b.data = device.move_tensor(b.data)
        self.cold = [g for g in self.groups if g.offloaded]
        self.hot = [g for g in self.groups if not g.offloaded]
        self.remote_groups = [g for g in self.groups if g.remote]
        for g in self.cold:
            self._handles.append(g.module.register_forward_pre_hook(self._make_pre_forward(g)))
            self._handles.append(g.module.register_forward_hook(self._make_post_forward(g)))
            for _, p in g.params:
                if p.requires_grad:
                    self._handles.append(p.register_post_accumulate_grad_hook(self._on_grad_accumulated))
        if self.device.backend == "cuda":
            self.device.synchronize()
        self._reset_counters()

    # ------------------------------------------------------------------ setup
    def _reserve_static(self, module, factory, offset, n) -> None:
        """Reserve everything that exists for the whole job, before materialising anything."""
        dev_res, ram_res = [], []
        sf, tf = OPTIMIZER_STATE_FACTOR[self.optimizer_name], OPTIMIZER_TEMP_FACTOR[self.optimizer_name]
        sizes = []
        count = len(module) if module is not None else n
        for local in range(count):
            idx = local + offset
            if module is not None:
                nbytes = sum(tensor_nbytes(p) for p in module[local].parameters())
            else:
                with torch.device("meta"):
                    nbytes = sum(tensor_nbytes(p) for p in factory(idx, meta=True).parameters())
            if nbytes:
                sizes.append((idx, nbytes, self.policy.is_cold(idx), self.policy.is_remote(idx)))
        hot = [s for s in sizes if not s[2]]
        cold = [s for s in sizes if s[2]]
        for idx, b, _, remote in hot:
            dev_res.append((group_id(idx) + ".grad", b, TensorRole.GRADIENT))
            if not self.cpu_opt:
                dev_res.append((group_id(idx) + ".optim", int(sf * b), TensorRole.OPTIMIZER_STATE))
        if hot and not self.cpu_opt:
            dev_res.append(("stage.optimizer_step_workspace", int(tf * sum(s[1] for s in hot)), TensorRole.TEMPORARY))
        if cold:
            biggest = max(s[1] for s in cold)
            dev_res.append(("offload.grad_workspace", biggest, TensorRole.TEMPORARY))
            if not self.cpu_opt:
                dev_res.append(("offload.optimizer_step_workspace", int(tf * biggest), TensorRole.TEMPORARY))
                for idx, b, _, _ in cold:
                    dev_res.append((group_id(idx) + ".optim", int(sf * b), TensorRole.OPTIMIZER_STATE))
        # Local RAM: gradient accumulators (COLD layers, or all layers with the CPU optimizer) and
        # CPU optimizer state of locally homed layers. Masters are registered as they are created.
        for idx, b, is_cold, remote in sizes:
            if is_cold or self.cpu_opt:
                ram_res.append((group_id(idx) + ".grad_ram", b, TensorRole.GRADIENT))
            if self.cpu_opt and not remote:
                ram_res.append((group_id(idx) + ".optim_ram", int(sf * b), TensorRole.OPTIMIZER_STATE))
        for tid, nbytes, role in dev_res:
            if nbytes:
                self.store.reserve_accelerator(nbytes, tid, None)
                self.store.register(tid, torch.empty(nbytes, dtype=torch.uint8, device="meta"), role, ACC)
        for tid, nbytes, role in ram_res:
            if nbytes:
                self.store.reserve_local(nbytes, tid, None)
                self.store.register(tid, torch.empty(nbytes, dtype=torch.uint8, device="meta"), role, RAM)
        remote_bytes = sum(int((1 + sf) * b) for _, b, _, r in sizes if r)
        if remote_bytes:
            self.store.reserve_remote(remote_bytes, "remote.reservation")
            self.client.reserve(int(remote_bytes * 1.02) + (1 << 20))
            self.remote_reserved = remote_bytes

    def _adopt(self, layer: nn.Module, idx: int) -> None:
        dev, store, pin = self.device, self.store, self.policy.pin_host_memory
        params = [(parameter_id(idx, n), p) for n, p in layer.named_parameters()]
        if not params:
            return
        g = ParameterGroup(group_id(idx), idx, layer, params, offloaded=self.policy.is_cold(idx),
                           remote=self.policy.is_remote(idx))
        self.groups.append(g)
        for tid, p in params:
            self.by_param[id(p)] = (g, tid, p)
            g.shapes[id(p)] = (tuple(p.shape), p.dtype)
        if g.remote:
            g.host_state = HostState.REMOTE_ONLY
            t0 = self._now()
            for tid, p in params:   # register version 1 remotely, keep nothing locally
                g.versions[tid] = self.client.put(tid, p.data.detach().contiguous().cpu(), old_version=-1,
                                                  new_version=1)
                self._count_put(tensor_nbytes(p), 0.0)
                store.register(tid, torch.empty(p.shape, dtype=p.dtype, device="meta"), TensorRole.PARAMETER, REMOTE,
                               group=g.group_id)
                store.locate(tid).version = g.versions[tid]
            self._span("REMOTE_PUT", t0, self._now(), g, what="initial placement")
            if not g.offloaded:   # HOT layer with a remote home: device copy lives for the whole job
                for tid, p in params:
                    store.reserve_accelerator(tensor_nbytes(p), tid, REMOTE)
                    p.data = dev.move_tensor(p.data)
                    store.set_residency(tid, REMOTE, (ACC,), dirty=False)
                    store.locate(tid).pinned = True
                g.state = GroupState.RESIDENT_ACCELERATOR
            else:
                for _, p in params:
                    p.data = torch.empty(0, dtype=p.dtype)
                g.state = GroupState.RESIDENT_RAM
            return
        for tid, p in params:
            if g.offloaded or self.cpu_opt:
                store.reserve_local(tensor_nbytes(p), tid, None)
                g.host[id(p)] = master = dev.host_copy(p.data, pinned=pin)
                store.pinned_host_bytes += tensor_nbytes(master) if master.is_pinned() else 0
            if g.offloaded:
                p.data = g.host[id(p)]
                store.register(tid, p, TensorRole.PARAMETER, RAM, group=g.group_id)
            else:
                store.reserve_accelerator(tensor_nbytes(p), tid, RAM)
                p.data = dev.move_tensor(p.data)
                meta = store.register(tid, p, TensorRole.PARAMETER, RAM if self.cpu_opt else ACC,
                                      group=g.group_id, pinned=True)
                if self.cpu_opt:
                    store.set_residency(tid, RAM, (ACC,), dirty=False)
                meta.pinned = True
        g.state = GroupState.RESIDENT_RAM if g.offloaded else GroupState.RESIDENT_ACCELERATOR

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        try:
            self._drain_writebacks()
        except Exception:
            pass
        if self.client is not None:
            try:
                self.client.release()
            except Exception:
                pass
            self.client.close()

    # --------------------------------------------------------------- counters
    _COUNTERS = ("prefetch_count", "prefetch_hits", "prefetch_late", "prefetch_misses", "prefetch_skipped_budget",
                 "resident_hits", "demand_loads", "eviction_count", "bytes_evicted", "writeback_count",
                 "writeback_bytes", "h2d_bytes", "d2h_bytes", "h2d_s", "d2h_s", "stall_s", "sync_load_s",
                 "optimizer_offload_s", "grad_offload_bytes", "pressure_evictions",
                 # V2.5 remote tier
                 "remote_fetch_count", "remote_fetch_bytes", "remote_fetch_network_s", "remote_fetch_queue_s",
                 "remote_fetch_total_s", "local_staging_s", "remote_fetch_exposed_s", "remote_prefetch_count",
                 "remote_prefetch_hits", "remote_prefetch_misses", "remote_writeback_count", "remote_writeback_bytes",
                 "remote_put_network_s", "remote_put_queue_s", "remote_commit_wait_s", "remote_layer_uses",
                 "staging_drops", "staging_pressure_drops", "layer_uses")

    def _reset_counters(self) -> None:
        for k, v in self.counters.items():
            self.totals[k] = self.totals.get(k, 0) + v
        self.counters = {k: 0 for k in self._COUNTERS}

    def _count_put(self, nbytes: int, network_s: float) -> None:
        self.counters["remote_writeback_bytes"] += nbytes
        self.counters["remote_put_network_s"] += network_s

    def _span(self, category: str, start: float, end: float, g: ParameterGroup | None = None, **args) -> None:
        tl = self.timeline
        if tl is not None:
            if g is not None:
                args.setdefault("group", g.group_id)
                args.setdefault("bytes", g.param_bytes)
            tl.add(category, start, end, self.current_step, None, **args)

    def _now(self) -> float:
        return self.timeline.now() if self.timeline is not None else time.time()

    # ------------------------------------------------------- local RAM staging
    def _make_room_local(self, nbytes: int, exclude: ParameterGroup | None) -> None:
        budget = self.store.local_ram_budget
        if budget is None:
            return
        while self.store.local_resident_bytes() + self.store.local_in_flight + nbytes > budget:
            self._collect_writebacks(block=False)
            if self.store.local_resident_bytes() + self.store.local_in_flight + nbytes <= budget:
                return
            idle = [g for g in self.remote_groups if g is not exclude and g.host_state == HostState.LOCAL_STAGED
                    and g.state == GroupState.RESIDENT_RAM and not g.in_window and not g.staged_state]
            if not idle:
                if self._writebacks:
                    self._collect_writebacks(block=True, at_most=1)
                    continue
                return   # reserve_local raises with the numbers
            victim = min(idle, key=lambda g: g.last_access)
            self._drop_staging(victim, reason="local RAM pressure")
            self.counters["staging_pressure_drops"] += 1

    def _alloc_staging(self, g: ParameterGroup, with_state: bool) -> tuple[list, list]:
        """Allocate (pinned) staging tensors for a remote group's params (+ AdamW state)."""
        pin = self.policy.pin_host_memory and self.device.supports("pinned_memory")
        keys = ADAM_KEYS if with_state and self._has_remote_state(g) else ()
        plan = []   # (tensor id, shape, dtype, is_param) of buffers that do not exist yet
        for tid, p in g.params:
            shape, dtype = g.shapes[id(p)]
            if g.host_state != HostState.LOCAL_STAGED or id(p) not in g.host:
                plan.append((tid, p, shape, dtype, None))
            for k in keys:
                plan.append((f"{tid}.optim.{k}", p, shape, dtype, k))
        need = sum(math_bytes(shape, dtype) for _, _, shape, dtype, _ in plan)   # exactly what is allocated
        self._make_room_local(need, g)
        self.store.reserve_local(need, g.params[0][0], REMOTE)
        t0 = self._now()
        targets, ids = [], []
        for tid, p, shape, dtype, k in plan:
            buf = torch.empty(shape, dtype=dtype, pin_memory=pin)
            if k is None:
                g.host[id(p)] = buf
            else:
                g.staged_state.setdefault(id(p), {})[k] = buf
            targets.append(buf)
            ids.append(tid)
        self.store.begin_local_transfer(need)
        self.counters["local_staging_s"] += self._now() - t0
        return ids, targets

    def _has_remote_state(self, g: ParameterGroup) -> bool:
        return any(f"{tid}.optim.{k}" in g.versions for tid, _ in g.params for k in ADAM_KEYS)

    def _start_remote_fetch(self, g: ParameterGroup, *, with_state: bool, prefetch: bool) -> bool:
        if g.host_state == HostState.LOCAL_STAGED and not (with_state and self._has_remote_state(g)
                                                           and not g.staged_state):
            return True
        if g.host_state not in (HostState.REMOTE_ONLY, HostState.LOCAL_STAGED):
            return g.host_state == HostState.FETCHING_REMOTE
        try:
            ids, targets = self._alloc_staging(g, with_state)
        except MemoryBudgetError:
            if prefetch:
                self.counters["prefetch_skipped_budget"] += 1
                return False
            raise
        from meshtrain.networking.tensor_server import TransferTiming, wrap_future

        versions = [g.versions[i] for i in ids]
        timing = TransferTiming()
        client = self.client

        def run():
            for tid, buf, ver in zip(ids, targets, versions):
                client.get(tid, buf, version=ver, timing=timing)
            return timing

        g.fetch_issue = self._now()
        g.fetch_with_state = with_state
        state_bytes = sum(t.numel() * t.element_size() for i, t in zip(ids, targets) if ".optim." in i)
        g.fetch = (wrap_future(run), sum(t.numel() * t.element_size() for t in targets), state_bytes)
        g.host_transition(HostState.FETCHING_REMOTE)
        if prefetch:
            self.counters["remote_prefetch_count"] += 1
        return True

    def _finish_remote_fetch(self, g: ParameterGroup, *, demand: bool) -> None:
        fut, nbytes, state_bytes = g.fetch
        t0 = self._now()
        late = not fut.done()
        try:
            timing = fut.result()
        except Exception as exc:
            raise self._remote_failure(exc, g) from exc
        t1 = self._now()
        g.fetch = None
        self.store.end_local_transfer(nbytes)
        self.store.add_local_extra(state_bytes)   # staged AdamW state (not a tracked record while staged)
        cached = (RAM,) if g.offloaded else (RAM, ACC)
        for tid, p in g.params:
            if RAM not in self.store.locate(tid).resident_tiers():
                self.store.set_residency(tid, REMOTE, cached, dirty=False)
        g.host_transition(HostState.LOCAL_STAGED)
        g.staged_at = t1
        c = self.counters
        c["remote_fetch_count"] += 1
        c["remote_fetch_bytes"] += timing.bytes
        c["remote_fetch_network_s"] += timing.network_s
        c["remote_fetch_queue_s"] += timing.queue_wait_s
        c["remote_fetch_total_s"] += t1 - g.fetch_issue
        if demand:
            c["remote_fetch_exposed_s"] += t1 - g.fetch_issue
            c["remote_prefetch_misses"] += 1
            self._span("REMOTE_GET", g.fetch_issue, t1, g, network_ms=timing.network_s * 1000,
                       queue_ms=timing.queue_wait_s * 1000, fetched=timing.bytes)
        else:
            c["remote_prefetch_hits"] += 1
            if late:
                c["remote_fetch_exposed_s"] += t1 - t0
                self._span("REMOTE_FETCH_STALL", t0, t1, g)
            self._span("REMOTE_PREFETCH", g.fetch_issue, t1 if late else t0, g,
                       network_ms=timing.network_s * 1000, queue_ms=timing.queue_wait_s * 1000, fetched=timing.bytes)

    def _ensure_staged(self, g: ParameterGroup, *, with_state: bool = False) -> None:
        if not g.remote:
            return
        if g.host_state == HostState.WRITEBACK_REMOTE:
            self._collect_writebacks(block=True, group=g)
        if g.host_state == HostState.LOCAL_STAGED and (not with_state or g.staged_state
                                                       or not self._has_remote_state(g)):
            return
        if g.host_state != HostState.FETCHING_REMOTE:
            self._start_remote_fetch(g, with_state=with_state, prefetch=False)
            self._finish_remote_fetch(g, demand=True)
        else:
            self._finish_remote_fetch(g, demand=False)
            if with_state and self._has_remote_state(g) and not g.staged_state:
                self._start_remote_fetch(g, with_state=True, prefetch=False)
                self._finish_remote_fetch(g, demand=True)

    def _drop_staging(self, g: ParameterGroup, reason: str = "") -> None:
        """Free a clean staged copy (the remote copy stays authoritative)."""
        if g.host_state != HostState.LOCAL_STAGED:
            return
        if g.state != GroupState.RESIDENT_RAM:
            raise RuntimeError(f"cannot drop staging of {g.group_id} while it is {g.state.value}")
        for tid, p in g.params:
            if self.store.locate(tid).dirty:
                raise RuntimeError(f"refusing to drop dirty staged copy of {tid}")
            self.store.set_residency(tid, REMOTE, ())
        for _, p in g.params:
            g.host.pop(id(p), None)
            p.data = torch.empty(0, dtype=g.shapes[id(p)][1])
        state_bytes = sum(t.numel() * t.element_size() for st in g.staged_state.values() for t in st.values())
        g.staged_state.clear()
        self.store.release_local(state_bytes)
        g.host_transition(HostState.REMOTE_ONLY)
        self.counters["staging_drops"] += 1
        if g.staged_at is not None:
            self._span("REMOTE_RESIDENCY", g.staged_at, self._now(), g, thread=f"staging {g.group_id}",
                       reason=reason)
            g.staged_at = None

    def _remote_failure(self, exc: Exception, g: ParameterGroup | None):
        from meshtrain.networking.tensor_server import RemoteMemoryError

        if isinstance(exc, RemoteMemoryError):
            affected = [x.group_id for x in self.remote_groups]
            msg = (f"remote memory failure on {exc.worker or (self.policy.remote.worker if self.policy.remote else '?')}"
                   f" ({exc.code}): {exc}. Affected tensors: {len(affected)} remote layer groups "
                   f"({', '.join(affected[:6])}{'...' if len(affected) > 6 else ''}). Residency: "
                   f"{self.diagnostics(short=True)}")
            err = RemoteMemoryError(msg, exc.code, exc.worker, exc.tensor_id)
            return err
        return exc

    # ------------------------------------------------------------ data movement
    def _make_room(self, nbytes: int, exclude: ParameterGroup) -> None:
        budget = self.store.accelerator_budget
        if budget is None:
            return
        while self.store.accelerator_resident_bytes() + self.store.in_flight_bytes + nbytes > budget:
            idle = [g for g in self.cold if g is not exclude and not g.in_window and g.state in
                    (GroupState.RESIDENT_ACCELERATOR, GroupState.SAVED_FOR_BACKWARD)]
            if not idle:
                return  # reserve_accelerator raises with the numbers
            victim = min(idle, key=lambda g: g.last_access)
            self.unload(victim, reason="memory pressure")
            self.counters["pressure_evictions"] += 1

    def _start_load(self, g: ParameterGroup, *, prefetch: bool) -> bool:
        need = g.param_bytes
        tid0 = g.params[0][0]
        if g.remote and g.host_state != HostState.LOCAL_STAGED:
            if prefetch:
                return False   # the remote stage of the prefetch pipeline has not delivered yet
            self._ensure_staged(g)
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
        auth = REMOTE if g.remote else RAM
        for (tid, p), t in zip(g.params, tensors):
            p.data = t
            self.store.set_residency(tid, auth, (RAM, ACC) if g.remote else (ACC,), dirty=False)
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
        rd = self.policy.remote_prefetch_distance
        rd = d + 1 if rd is None else rd
        order = self.cold if direction > 0 else self.cold[::-1]
        i = order.index(g)
        if rd > 0 and self.client is not None:   # stage 1: remote RAM -> local staging
            for nxt in order[i + 1:i + 1 + rd]:
                if nxt.remote and nxt.host_state == HostState.REMOTE_ONLY:
                    self._start_remote_fetch(nxt, with_state=False, prefetch=True)
        if d <= 0:
            return
        for nxt in order[i + 1:i + 1 + d]:         # stage 2: local RAM -> device
            if nxt.remote and nxt.host_state == HostState.FETCHING_REMOTE and nxt.fetch[0].done():
                self._finish_remote_fetch(nxt, demand=False)
            if nxt.state == GroupState.RESIDENT_RAM and self._start_load(nxt, prefetch=True):
                self.counters["prefetch_count"] += 1

    def ensure_loaded(self, g: ParameterGroup, phase: str) -> None:
        """Make ``g`` usable on the device for ``phase`` (forward | backward | optimizer | manual)."""
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
            auth = REMOTE if g.remote else RAM
            for tid, p in g.params:
                if self.store.locate(tid).dirty:
                    raise RuntimeError(f"refusing to drop dirty {tid}")
                p.data = g.host[id(p)]
                self.store.set_residency(tid, auth, (RAM,) if g.remote else ())
            g.transition(GroupState.RESIDENT_RAM)
            self.counters["eviction_count"] += 1
            self.counters["bytes_evicted"] += g.param_bytes
            t1 = self._now()
            self._span("TENSOR_EVICT", t0, t1, g, reason=reason)
            if g.loaded_at is not None:
                self._span("TENSOR_ACCELERATOR_RESIDENT", g.loaded_at, t1, g, thread=f"residency {g.group_id}")
                g.loaded_at = None
            if g.remote and not self.policy.reuse:
                self._drop_staging(g, reason="no reuse")   # reuse off: every use refetches

    # ------------------------------------------------------------------- hooks
    def _make_pre_forward(self, g: ParameterGroup):
        def hook(module, args):
            self.counters["layer_uses"] += 1
            if g.remote:
                self.counters["remote_layer_uses"] += 1
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
                if self.window_mode:
                    return     # layer-major: evicted once by end_window after all microbatches
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
            if g.group_id + ".grad_ram" in self.store:
                self.store.discard(g.group_id + ".grad_ram")   # startup reservation -> real accumulators
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
            self.store.add_local_extra(tensor_nbytes(buf))
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
        self.counters["layer_uses"] += 1
        if g.remote:
            self.counters["remote_layer_uses"] += 1
        if g.state == GroupState.IN_USE_BACKWARD:
            g.transition(GroupState.SAVED_FOR_BACKWARD if g.pending_backward else GroupState.RESIDENT_ACCELERATOR)
        if self.window_mode:
            return
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

    # ------------------------------------------------- layer-major reuse windows
    def begin_window(self, layer: nn.Module) -> None:
        g = self._group_of(layer)
        if g is not None:
            g.in_window = True

    def end_window(self, layer: nn.Module, phase: str) -> None:
        """All microbatches of one pass are done with ``layer``: release it once."""
        g = self._group_of(layer)
        if g is None or not g.offloaded:
            if g is not None:
                g.in_window = False
            return
        with self.lock:
            g.in_window = False
            if g.state == GroupState.IN_USE_BACKWARD:
                g.transition(GroupState.SAVED_FOR_BACKWARD if g.pending_backward else GroupState.RESIDENT_ACCELERATOR)
            if g.state in _LOADED and g.state not in (GroupState.IN_USE_FORWARD, GroupState.IN_USE_BACKWARD):
                self.unload(g, reason=f"end of {phase} window")
            if g.remote and phase == "forward" and self.store.local_ram_budget is not None:
                pass  # staged copy kept while it fits; dropped under local RAM pressure (LRU)

    def _group_of(self, layer: nn.Module) -> ParameterGroup | None:
        for g in self.groups:
            if g.module is layer:
                return g
        return None

    # --------------------------------------------------------------- optimizer
    def gradient(self, p: nn.Parameter) -> torch.Tensor | None:
        g = self.by_param[id(p)][0]
        if g.offloaded or (self.cpu_opt and id(p) in g.host_grad and p.grad is None):
            return g.host_grad[id(p)] if id(p) in g.grad_valid else None
        return p.grad

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        with self.lock:
            for g in self.cold:  # nothing may still be referenced by an unfinished microbatch
                g.pending_backward = 0
                g.in_window = False
                if g.state in _LOADED:
                    if g.state in (GroupState.IN_USE_FORWARD, GroupState.IN_USE_BACKWARD,
                                   GroupState.SAVED_FOR_BACKWARD):
                        g.transition(GroupState.RESIDENT_ACCELERATOR)
                    self.unload(g, reason="before optimizer step")
                elif g.state == GroupState.PREFETCHING:
                    self.unload(g, reason="before optimizer step")
            try:
                if self.cpu_opt:
                    self._cpu_step(optimizer)
                else:
                    self._accelerator_step(optimizer)
            except Exception as exc:
                raise self._remote_failure(exc, None) from exc
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
        """One layer at a time: masters (+ state) in RAM, CPU AdamW, refresh HOT copies, write remote back."""
        t_all = self._now()
        order = list(self.groups)
        # Every gradient must be in RAM and detached before the per-layer steps: optimizer.step() updates
        # every parameter that has a .grad, so a HOT layer's device gradient would otherwise be applied
        # while an earlier layer steps (and again in its own turn).
        for g in self.hot:
            for tid, p in g.params:
                if p.grad is not None:
                    self._accumulate_host_grad(g, tid, p, p.grad)
                    p.grad = None
        for g in self.groups:
            for _, p in g.params:
                p.grad = None
        rd = max(1, self.policy.remote_prefetch_distance if self.policy.remote_prefetch_distance is not None
                 else self.policy.prefetch_distance + 1)
        for i, g in enumerate(order):
            # stage the next remote layers (master + AdamW state) while this one steps
            for nxt in order[i + 1:i + 1 + rd]:
                if nxt.remote and nxt.host_state == HostState.REMOTE_ONLY:
                    self._start_remote_fetch(nxt, with_state=True, prefetch=True)
            if g.remote:
                self._ensure_staged(g, with_state=True)
                self.counters["remote_layer_uses"] += 1
            t0 = self._now()
            device_copies = {}
            for tid, p in g.params:
                if not g.offloaded and p.grad is not None:   # HOT layer: gradients on the device
                    self._accumulate_host_grad(g, tid, p, p.grad)
                    p.grad = None
                if not g.offloaded:
                    device_copies[id(p)] = p.data
                p.data = g.host[id(p)]
                p.grad = g.host_grad[id(p)] if id(p) in g.grad_valid else None
                if g.remote:
                    st = optimizer.state[p]
                    for k, v in g.staged_state.get(id(p), {}).items():
                        st[k] = v
            optimizer.step()        # only this layer's parameters have gradients
            for tid, p in g.params:
                p.grad = None
            t1 = self._now()
            if not g.offloaded:     # refresh the device working copy from the updated master
                for tid, p in g.params:
                    dev_t = device_copies[id(p)]
                    dev_t.copy_(g.host[id(p)], non_blocking=g.host[id(p)].is_pinned())
                    p.data = dev_t
                self.counters["h2d_bytes"] += g.param_bytes
            elif g.remote:
                for _, p in g.params:
                    p.data = g.host[id(p)]
            for tid, _ in g.params:
                meta = self.store.locate(tid)
                if g.remote:
                    self.store.mark_dirty(tid, RAM)   # staged master is newer than REMOTE_RAM
                    if not g.offloaded:               # ... and the device copy was refreshed from it
                        self.store.set_residency(tid, RAM, (ACC,), dirty=True)
                else:
                    meta.version += 1
            if g.remote:
                self._start_remote_writeback(g, optimizer)
            self.counters["optimizer_offload_s"] += t1 - t0
        self._drain_writebacks()
        self.device.synchronize()
        self._span("OPTIMIZER_OFFLOAD", t_all, self._now(), what="CPU optimizer step (all layers)")

    def _start_remote_writeback(self, g: ParameterGroup, optimizer) -> None:
        from meshtrain.networking.tensor_server import TransferTiming, wrap_future

        items = []
        for tid, p in g.params:
            items.append((tid, g.host[id(p)]))
            st = optimizer.state[p]
            g.staged_state[id(p)] = {k: st[k] for k in ADAM_KEYS if k in st}
            for k in ADAM_KEYS:
                if k in st:
                    items.append((f"{tid}.optim.{k}", st[k]))
                    st.pop(k)          # state lives remotely; restored from staging before the next step
        if g.host_state == HostState.LOCAL_STAGED:
            g.host_transition(HostState.WRITEBACK_REMOTE)
        timing = TransferTiming()
        old = {tid: g.versions.get(tid, -1) for tid, _ in items}
        client = self.client

        def run():
            committed = {}
            for tid, t in items:
                committed[tid] = client.put(tid, t, old_version=old[tid], new_version=max(old[tid], 0) + 1,
                                            timing=timing)
            return committed, timing

        self.counters["remote_writeback_count"] += 1
        self._writebacks.append((g, wrap_future(run), self._now(), {tid: t for tid, t in items}))
        while len(self._writebacks) > max(1, self.policy.remote.max_inflight_writebacks if self.policy.remote else 2):
            self._collect_writebacks(block=True, at_most=1)

    def _collect_writebacks(self, block: bool, at_most: int | None = None, group: ParameterGroup | None = None) -> None:
        done = 0
        remaining = []
        for item in self._writebacks:
            g, fut, issued, tensors = item
            if (group is not None and g is not group) or (at_most is not None and done >= at_most):
                remaining.append(item)
                continue
            if not block and not fut.done():
                remaining.append(item)
                continue
            t0 = self._now()
            try:
                committed, timing = fut.result()
            except Exception as exc:
                self._writebacks = remaining + [x for x in self._writebacks if x is not item and x not in remaining]
                raise self._remote_failure(exc, g) from exc
            t1 = self._now()
            done += 1
            cached = (RAM,) if g.offloaded else (RAM, ACC)
            for tid, ver in committed.items():
                g.versions[tid] = ver
                if tid in self.store and self.store.locate(tid).role == TensorRole.PARAMETER:
                    self.store.set_residency(tid, REMOTE, cached, dirty=False)
                    self.store.locate(tid).version = ver
                elif tid in self.store:
                    self.store.set_residency(tid, REMOTE, (), dirty=False)
                    self.store.locate(tid).version = ver
                else:   # first commit of an optimizer-state tensor (record only: no local reference kept)
                    t = tensors[tid]
                    self.store.register(tid, torch.empty(t.shape, dtype=t.dtype, device="meta"),
                                        TensorRole.OPTIMIZER_STATE, REMOTE, group=g.group_id)
                    self.store.locate(tid).version = ver
            c = self.counters
            c["remote_put_network_s"] += timing.network_s
            c["remote_put_queue_s"] += timing.queue_wait_s
            c["remote_writeback_bytes"] += timing.bytes
            c["remote_commit_wait_s"] += t1 - t0
            self._span("REMOTE_PUT", issued, t1, g, network_ms=timing.network_s * 1000, written=timing.bytes)
            g.host_transition(HostState.LOCAL_STAGED)
            if g.offloaded and g.state == GroupState.RESIDENT_RAM:
                self._drop_staging(g, reason="committed")
            elif not g.offloaded:   # HOT layer: master staged only for the step
                self._drop_hot_staging(g)
        self._writebacks = remaining

    def _drop_hot_staging(self, g: ParameterGroup) -> None:
        for tid, p in g.params:
            g.host.pop(id(p), None)
            self.store.set_residency(tid, REMOTE, (ACC,))
        state_bytes = sum(t.numel() * t.element_size() for st in g.staged_state.values() for t in st.values())
        g.staged_state.clear()
        self.store.release_local(state_bytes)
        g.host_transition(HostState.REMOTE_ONLY)

    def _drain_writebacks(self) -> None:
        while self._writebacks:
            self._collect_writebacks(block=True)

    def zero_grad(self) -> None:
        for g in self.groups:
            g.grad_valid.clear()

    def refresh_records(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Register real optimizer-state tensors of local layers (replacing startup reservations)."""
        store = self.store
        if optimizer is None:
            return
        state_tier = RAM if self.cpu_opt else ACC
        for g in self.groups:
            if g.remote:
                continue
            for tid, p in g.params:
                for key, v in optimizer.state.get(p, {}).items():
                    if torch.is_tensor(v) and v.dim() > 0:
                        for reserved in (g.group_id + ".optim", g.group_id + ".optim_ram"):
                            if reserved in store:
                                store.discard(reserved)
                        store.register(f"{tid}.optim.{key}", v, TensorRole.OPTIMIZER_STATE, state_tier,
                                       group=g.group_id)

    # ----------------------------------------------------------------- reports
    def fetch_logical(self, g: ParameterGroup) -> dict[str, torch.Tensor]:
        """Authoritative values of a remote group's params and AdamW state (checkpoints)."""
        self._drain_writebacks()
        out = {}
        for tid, p in g.params:
            if id(p) in g.host and g.host_state in (HostState.LOCAL_STAGED,):
                out[tid] = g.host[id(p)].clone()
            else:
                shape, dtype = g.shapes[id(p)]
                buf = torch.empty(shape, dtype=dtype)
                self.client.get(tid, buf, version=g.versions[tid])
                out[tid] = buf
            for k in ADAM_KEYS:
                sid = f"{tid}.optim.{k}"
                if sid in g.versions:
                    shape, dtype = g.shapes[id(p)]
                    buf = torch.empty(shape, dtype=dtype)
                    self.client.get(sid, buf, version=g.versions[sid])
                    out[sid] = buf
        return out

    def logical_value(self, p: nn.Parameter) -> torch.Tensor:
        """CPU copy of a parameter's authoritative value, wherever it lives (fetches remote layers)."""
        g, tid, _ = self.by_param[id(p)]
        if g.remote and p.numel() == 0:
            self._drain_writebacks()
            shape, dtype = g.shapes[id(p)]
            buf = torch.empty(shape, dtype=dtype)
            self.client.get(tid, buf, version=g.versions[tid])
            return buf
        return p.detach().cpu().clone()

    def store_logical(self, g: ParameterGroup, values: dict[str, torch.Tensor]) -> None:
        """Overwrite a remote group's authoritative params / AdamW state (checkpoint load)."""
        self._drain_writebacks()
        if g.host_state == HostState.LOCAL_STAGED and g.state == GroupState.RESIDENT_RAM:
            self._drop_staging(g, reason="checkpoint load")
        for tid, t in values.items():
            old = g.versions.get(tid, -1)
            g.versions[tid] = self.client.put(tid, t.contiguous(), old_version=old, new_version=max(old, 0) + 1)
            if tid in self.store:
                self.store.locate(tid).version = g.versions[tid]

    def step_begin(self, step: int) -> None:
        self.current_step = step
        self.store.reset_peaks()

    def step_stats(self) -> dict:
        c = dict(self.counters)
        lookups = c["resident_hits"] + c["prefetch_hits"] + c["demand_loads"]
        st = self.store.stats()
        n_remote = len(self.remote_groups)
        out = {
            "strategy": self.policy.strategy, "optimizer_execution": self.policy.optimizer_execution,
            "prefetch_distance": self.policy.prefetch_distance, "eviction": self.policy.eviction,
            "reuse": self.policy.reuse,
            "groups_hot": len(self.hot), "groups_cold": len(self.cold), "groups_remote": n_remote,
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
            "local_ram_budget": st["local_ram_budget"], "local_ram_peak": st["local_peak"],
            "remote_resident": st["remote_resident"], "remote_peak": st["remote_peak"],
            "pinned_bytes": st["pinned_bytes"], "requested_budget": st["requested_budget"],
            "headroom": st["headroom"], "in_flight_peak": st["in_flight_peak"],
            "budget_violations": st["budget_violations"],
            "layer_uses": c["layer_uses"],
        }
        if n_remote:
            uses = c["remote_layer_uses"]
            out.update({
                "remote_fetch_count": c["remote_fetch_count"], "remote_bytes_read": c["remote_fetch_bytes"],
                "remote_writeback_count": c["remote_writeback_count"], "remote_bytes_written": c["remote_writeback_bytes"],
                "remote_fetch_network_ms": c["remote_fetch_network_s"] * 1000,
                "remote_queue_wait_ms": c["remote_fetch_queue_s"] * 1000,
                "local_staging_ms": c["local_staging_s"] * 1000,
                "total_remote_fetch_ms": c["remote_fetch_total_s"] * 1000,
                "exposed_remote_fetch_ms": c["remote_fetch_exposed_s"] * 1000,
                "hidden_remote_fetch_ms": max(0.0, c["remote_fetch_total_s"] - c["remote_fetch_exposed_s"]) * 1000,
                "remote_put_network_ms": c["remote_put_network_s"] * 1000,
                "remote_put_queue_ms": c["remote_put_queue_s"] * 1000,
                "remote_commit_wait_ms": c["remote_commit_wait_s"] * 1000,
                "remote_prefetch_count": c["remote_prefetch_count"],
                "remote_prefetch_hits": c["remote_prefetch_hits"], "remote_prefetch_misses": c["remote_prefetch_misses"],
                "remote_layer_uses": uses,
                "remote_reuse_ratio": (1.0 - c["remote_fetch_count"] / uses) if uses else 0.0,
                "remote_fetches_per_layer": c["remote_fetch_count"] / n_remote,
                "staging_drops": c["staging_drops"], "staging_pressure_drops": c["staging_pressure_drops"],
            })
        self._reset_counters()
        return out

    def diagnostics(self, short: bool = False) -> str:
        parts = []
        for g in self.groups:
            if short and not (g.offloaded or g.remote):
                continue
            s = f"{g.group_id}:{g.state.value}"
            if g.remote:
                s += f"/{g.host_state.value}"
                if g.versions:
                    s += f"/v{max(g.versions.values())}"
            parts.append(s)
        pending = [g.group_id for g, *_ in self._writebacks]
        fetching = [g.group_id for g in self.remote_groups if g.host_state == HostState.FETCHING_REMOTE]
        return (f"step {self.current_step}; fetching {fetching}; writebacks {pending}; "
                f"groups [{'; '.join(parts[:24])}{' ...' if len(parts) > 24 else ''}]")

    def audit(self) -> dict:
        """Recompute residency from the tensors themselves (tests: ledger == reality)."""
        on_device = 0
        problems = []
        for g in self.groups:
            for tid, p in g.params:
                meta = self.store.locate(tid)
                host = g.host.get(id(p))
                on_host_master = host is not None and p.data_ptr() == host.data_ptr()
                placeholder = p.numel() == 0
                dev_resident = (p.data.device.type != "cpu") or (
                    self.device.backend == "cpu" and g.offloaded and not on_host_master and not placeholder)
                if g.offloaded:
                    if dev_resident != (ACC in meta.resident_tiers()):
                        problems.append(f"{tid}: ledger {meta.resident_tiers()} vs device={dev_resident}")
                    if not dev_resident and not on_host_master and not (g.remote and placeholder):
                        problems.append(f"{tid}: evicted but not pointing at its host copy")
                if g.remote and meta.tier != REMOTE and not meta.dirty:
                    problems.append(f"{tid}: remote-homed but authoritative tier {meta.tier.value}")
                if dev_resident:
                    on_device += tensor_nbytes(p)
        return {"parameter_bytes_on_device": on_device, "problems": problems}

"""Validated job configuration (YAML -> pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobConfig(_Strict):
    name: str = "meshtrain-job"
    seed: int = 0
    runs_dir: str = "runs"


class ModelConfig(_Strict):
    type: Literal["mlp", "tiny_transformer"]
    # mlp
    sizes: list[int] | None = None
    # tiny_transformer
    layers: int = Field(6, ge=1)
    hidden_size: int = Field(256, ge=8)
    heads: int = Field(4, ge=1)
    vocab_size: int = Field(256, ge=2)
    seq_len: int = Field(64, ge=2)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    @model_validator(mode="after")
    def _check(self):
        if self.type == "tiny_transformer" and self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.sizes is not None and (len(self.sizes) < 2 or min(self.sizes) < 1):
            raise ValueError("mlp sizes need >= 2 positive entries")
        return self

    def spec_kwargs(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if self.type == "mlp":
            return {k: v for k, v in d.items() if k in ("type", "sizes", "dtype")}
        return {k: v for k, v in d.items() if k != "sizes"}


class TrainingConfig(_Strict):
    batch_size: int = Field(32, ge=1)
    microbatch_size: int | None = Field(None, ge=1)
    learning_rate: float = Field(0.01, gt=0)
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    steps: int = Field(100, ge=1)
    log_every: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _check(self):
        mb = self.microbatch_size or self.batch_size
        if self.batch_size % mb:
            raise ValueError("batch_size must be a multiple of microbatch_size")
        return self

    @property
    def num_microbatches(self) -> int:
        return self.batch_size // (self.microbatch_size or self.batch_size)


class StagePlacement(_Strict):
    worker: str | None = None  # worker id/hostname, or None for local runs
    device: str | None = None  # cpu|cuda|mps, local runs only
    layers: tuple[int, int]    # [start, end) layer range


class PlacementConfig(_Strict):
    # topology_aware (= auto): memory + compute + D2H/network/H2D + overlap + boundary size
    strategy: Literal["topology_aware", "auto", "equal", "compute", "manual"] = "topology_aware"
    num_stages: int | None = Field(None, ge=1)  # None = use all eligible workers
    stages: list[StagePlacement] | None = None   # required for manual
    # Reject plans whose memory estimate exceeds a worker's budget. Disable only to
    # probe real out-of-memory limits (capacity experiment, hardware mode).
    enforce_memory_check: bool = True
    # Deprecated V1 margin; used only when set explicitly (otherwise see ``memory:``).
    memory_headroom_fraction: float | None = Field(None, ge=0, lt=1)
    memory_headroom_min_gb: float | None = Field(None, ge=0)
    # Safety multiplier on the measured autograd-saved bytes (planner/memory.py).
    activation_overhead_factor: float = Field(1.25, ge=1)

    @model_validator(mode="after")
    def _check(self):
        if self.strategy == "manual" and not self.stages:
            raise ValueError("manual placement requires placement.stages")
        if self.stages:
            prev_end = None
            for s in self.stages:
                if s.layers[0] >= s.layers[1]:
                    raise ValueError(f"empty layer range {s.layers}")
                if prev_end is not None and s.layers[0] != prev_end:
                    raise ValueError("manual stages must be contiguous and ordered")
                prev_end = s.layers[1]
        return self


class RemoteRamConfig(_Strict):
    """V2.5: RAM on another machine as a backing tier (networking/tensor_server.py)."""

    enabled: bool = False
    budget_mb: float | None = Field(None, gt=0)   # this job's remote RAM budget (reserved up front)
    max_bytes: int | None = Field(None, gt=0)      # same, in bytes (budget_mb wins when both are set)
    workers: list[str] = []            # explicit tensor-service addresses "host:port"
    preferred_workers: list[str] = []  # worker names resolved through the coordinator
    max_inflight_fetches: int = Field(2, ge=1)
    max_inflight_writebacks: int = Field(2, ge=1)
    checksum: Literal["none", "crc32"] = "none"
    timeout_s: float = Field(120.0, gt=0)
    emulate_bandwidth_mbps: float | None = Field(None, gt=0)   # experiments only: throttle the link
    emulate_latency_ms: float = Field(0.0, ge=0)

    def budget_bytes(self) -> int | None:
        if self.budget_mb is not None:
            return int(self.budget_mb * 1024**2)
        return self.max_bytes


class ReuseConfig(_Strict):
    # layer-major execution on single-stage jobs: load each offloaded layer once per pass
    # for all microbatches instead of once per microbatch (V2.5)
    enabled: bool = False


class PrefetchConfig(_Strict):
    enabled: bool = True
    distance: int | None = Field(None, ge=0)        # overrides memory.prefetch_distance
    remote_distance: int | None = Field(None, ge=0)  # remote RAM -> local staging look-ahead


class MemoryConfig(_Strict):
    """Planner safety margins and runtime memory validation (planner/memory.py)."""

    # Fraction of detected device memory the planner may use.
    safety_factor: float = Field(0.85, gt=0, le=1)
    # Per-backend overrides of safety_factor (unified-memory MPS shares RAM with the OS).
    backend_safety_factor: dict[str, float] = {"mps": 0.80}
    # Memory the framework itself takes before any tensor (CUDA context, cuBLAS workspace).
    framework_reserve_gb: dict[str, float] = {"cuda": 0.4, "mps": 0.0, "cpu": 0.25}
    validate_runtime_usage: bool = True   # measure actual memory after materialising a stage
    probe_allocation: bool = True         # accelerators: allocate the estimated remainder once at startup
    max_replans: int = Field(2, ge=0)     # startup OOM -> replan with the measured budget, at most N times
    replan_shrink: float = Field(0.85, gt=0, lt=1)  # extra margin applied to a failed worker's budget

    # -- V2 tensor residency (runtime/offload.py, docs/v2-architecture.md) --------
    # static: V1.5 behaviour, every tensor stays on the stage's device.
    # manual_offload: layers not listed in keep_resident live in local RAM between uses.
    # auto_offload: planner/residency.py chooses which layers stay resident.
    strategy: Literal["static", "manual_offload", "auto_offload", "remote_offload"] = "static"
    # Accelerator budget per stage: total bytes this process may allocate on its device.
    # accelerator_budget_mb takes precedence; accelerator_budget accepts "auto", "6GB", "512MB".
    accelerator_budget_mb: float | None = Field(None, gt=0)
    accelerator_budget: str | None = None
    # CUDA: also cap the caching allocator at the budget (real OOM beyond it, reproducible tests).
    enforce_allocator_limit: bool = True
    keep_resident: list[str] = []       # manual_offload: layers that stay on the device, e.g. "layers.0"
    prefetch_distance: int = Field(1, ge=0)   # 0 = synchronous loads; N = load N offloaded layers ahead
    # after_use: evict a layer after each forward/backward use (lowest memory).
    # after_backward: keep a loaded layer until its last pending backward (conservative).
    eviction: Literal["after_use", "after_backward"] = "after_use"
    pin_host_memory: bool = True        # page-locked host copies (CUDA)
    use_local_ram: bool = True
    use_remote_ram: bool = False        # alias for remote_ram.enabled
    optimizer_offload: bool | None = None  # true = optimizer.execution cpu_offload
    local_ram_budget_mb: float | None = Field(None, gt=0)   # host RAM MeshTrain may use for model state
    remote_ram: RemoteRamConfig = RemoteRamConfig()
    reuse: ReuseConfig = ReuseConfig()
    prefetch: PrefetchConfig = PrefetchConfig()

    @model_validator(mode="after")
    def _check_residency(self):
        if self.use_remote_ram:
            self.remote_ram.enabled = True
        if self.strategy == "remote_offload" and not self.remote_ram.enabled:
            raise ValueError("memory.strategy remote_offload needs memory.remote_ram.enabled: true")
        if self.remote_ram.enabled and self.strategy != "remote_offload":
            raise ValueError("memory.remote_ram is used only with memory.strategy: remote_offload")
        if self.strategy != "static" and not self.use_local_ram:
            raise ValueError(f"memory.strategy {self.strategy} needs use_local_ram: true")
        if self.accelerator_budget is not None and self.accelerator_budget_mb is None:
            parse_bytes(self.accelerator_budget)  # validate
        return self

    def budget_bytes(self, device_total: int | None = None, backend: str = "cuda") -> int | None:
        """Resolved accelerator budget in bytes (None = unlimited / device size)."""
        if self.accelerator_budget_mb is not None:
            return int(self.accelerator_budget_mb * 1024**2)
        if self.accelerator_budget in (None, ""):
            return None
        if self.accelerator_budget.strip().lower() == "auto":
            if device_total is None:
                return None
            sf = self.safety_factors().get(backend, self.safety_factor)
            return max(0, int(device_total * sf) - self.framework_reserve_bytes().get(backend, 0))
        return parse_bytes(self.accelerator_budget)

    def safety_factors(self) -> dict[str, float]:
        out = {b: self.safety_factor for b in ("cuda", "mps", "cpu")}
        out.update(self.backend_safety_factor)
        return out

    def framework_reserve_bytes(self) -> dict[str, int]:
        return {b: int(v * 1024**3) for b, v in self.framework_reserve_gb.items()}


def parse_bytes(text: str) -> int:
    """'6GB' / '512 MB' / '1.5GiB' / '2048' (MB) -> bytes (binary units)."""
    s = str(text).strip().upper().replace(" ", "").replace("IB", "B")
    for suffix, mult in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if s.endswith(suffix):
            try:
                return int(float(s[: -len(suffix)]) * mult)
            except ValueError:
                break
    try:
        return int(float(s) * 1024**2)
    except ValueError:
        raise ValueError(f"cannot parse memory size {text!r}; use e.g. 6GB or 512MB") from None


class OptimizerConfig(_Strict):
    # accelerator: V1.5 behaviour, optimizer state and update on the stage's device.
    # cpu_offload: optimizer state + fp32 master parameters in host RAM, update on the CPU (V2.3).
    execution: Literal["accelerator", "cpu_offload"] = "accelerator"


class WorkersConfig(_Strict):
    allow: list[Literal["cuda", "mps", "cpu"]] = ["cuda", "mps", "cpu"]
    require: list[str] = []  # worker ids that must take part


class NetworkConfig(_Strict):
    tensor_transport: Literal["tcp", "pipe"] = "tcp"
    timeout_s: float = Field(60.0, gt=0)
    connect_timeout_s: float = Field(30.0, gt=0)
    max_tensor_mb: float = Field(1024.0, gt=0)


class PipelineConfig(_Strict):
    schedule: Literal["gpipe", "1f1b"] = "1f1b"
    # Cap on microbatch graphs a stage may hold at once (1f1b); None = pipeline depth.
    max_inflight_microbatches: int | None = Field(None, ge=1)


class TransportConfig(_Strict):
    # async: sends run on a per-link sender thread (overlap with compute).
    # false reproduces V1 blocking sends (kept as a baseline).
    async_: bool = Field(True, alias="async")
    pinned_memory: bool = True   # CUDA: page-locked staging buffers
    buffer_pool: bool = True     # reuse staging buffers across microbatches
    max_outbound_queue: int | None = Field(None, ge=1)  # messages per link; default 2*M+4

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MeshTrainConfig(_Strict):
    job: JobConfig = JobConfig()
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    placement: PlacementConfig = PlacementConfig()
    workers: WorkersConfig = WorkersConfig()
    network: NetworkConfig = NetworkConfig()
    pipeline: PipelineConfig = PipelineConfig()
    transport: TransportConfig = TransportConfig()
    memory: MemoryConfig = MemoryConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    @model_validator(mode="after")
    def _check(self):
        if self.placement.stages:
            n = self.model_num_layers()
            if self.placement.stages[0].layers[0] != 0 or self.placement.stages[-1].layers[1] != n:
                raise ValueError(f"manual stages must cover layers [0, {n})")
        if self.memory.optimizer_offload is True or self.memory.strategy == "remote_offload":
            self.optimizer.execution = "cpu_offload"
        elif self.memory.optimizer_offload is False and self.optimizer.execution == "cpu_offload":
            raise ValueError("memory.optimizer_offload: false contradicts optimizer.execution: cpu_offload")
        return self

    def residency_policy(self):
        """runtime.offload.ResidencyPolicy for this config (static = V1.5)."""
        from meshtrain.runtime.offload import ResidencyPolicy

        m = self.memory
        distance = m.prefetch.distance if m.prefetch.distance is not None else m.prefetch_distance
        if not m.prefetch.enabled:
            distance = 0
        return ResidencyPolicy(strategy=m.strategy, keep_resident=tuple(m.keep_resident),
                               prefetch_distance=distance,
                               remote_prefetch_distance=(0 if not m.prefetch.enabled else m.prefetch.remote_distance),
                               eviction=m.eviction,
                               optimizer_execution=self.optimizer.execution, pin_host_memory=m.pin_host_memory,
                               enforce_allocator_limit=m.enforce_allocator_limit,
                               budget_mb=m.accelerator_budget_mb, budget_spec=m.accelerator_budget,
                               local_ram_budget_mb=m.local_ram_budget_mb, reuse=m.reuse.enabled)

    def remote_spec(self, job_id: str, address: str, worker: str = "", token: str | None = None):
        """runtime.offload.RemoteSpec for the remote RAM worker at ``address`` (host:port)."""
        from meshtrain.runtime.offload import RemoteSpec

        r = self.memory.remote_ram
        return RemoteSpec(address=address, worker=worker or address, token=token, job_id=job_id,
                          budget_bytes=r.budget_bytes(), max_inflight_fetches=r.max_inflight_fetches,
                          max_inflight_writebacks=r.max_inflight_writebacks, checksum=r.checksum,
                          timeout_s=r.timeout_s,
                          emulate_bandwidth_Bps=r.emulate_bandwidth_mbps * 1e6 / 8 if r.emulate_bandwidth_mbps else None,
                          emulate_latency_s=r.emulate_latency_ms / 1000.0)

    def pipeline_settings(self, job_id: str, **overrides):
        """PipelineSettings for this config (shared by workers and local runs)."""
        from meshtrain.runtime.pipeline import PipelineSettings

        kw = dict(job_id=job_id, steps=self.training.steps, batch_size=self.training.batch_size,
                  num_microbatches=self.training.num_microbatches, timeout_s=self.network.timeout_s,
                  log_every=self.training.log_every, schedule=self.pipeline.schedule,
                  max_inflight_microbatches=self.pipeline.max_inflight_microbatches,
                  async_transport=self.transport.async_, max_outbound_queue=self.transport.max_outbound_queue,
                  pinned_memory=self.transport.pinned_memory, buffer_pool=self.transport.buffer_pool)
        kw.update(overrides)
        return PipelineSettings(**kw)

    def model_num_layers(self) -> int:
        from meshtrain.models import build_model_spec

        return build_model_spec(self.model.spec_kwargs(), seed=self.job.seed).num_layers

    def build_model_spec(self):
        from meshtrain.models import build_model_spec

        return build_model_spec(self.model.spec_kwargs(), seed=self.job.seed)


class ConfigError(ValueError):
    pass


def parse_config(data: dict) -> MeshTrainConfig:
    try:
        return MeshTrainConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid MeshTrain config:\n{exc}") from exc


def load_config(path: str | Path) -> MeshTrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return parse_config(data)

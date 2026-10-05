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
    strategy: Literal["auto", "equal", "compute", "manual"] = "auto"
    num_stages: int | None = Field(None, ge=1)  # None = use all eligible workers
    stages: list[StagePlacement] | None = None   # required for manual
    memory_headroom_fraction: float = Field(0.15, ge=0, lt=1)
    memory_headroom_min_gb: float = Field(0.5, ge=0)
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


class WorkersConfig(_Strict):
    allow: list[Literal["cuda", "mps", "cpu"]] = ["cuda", "mps", "cpu"]
    require: list[str] = []  # worker ids that must take part


class NetworkConfig(_Strict):
    tensor_transport: Literal["tcp", "pipe"] = "tcp"
    timeout_s: float = Field(60.0, gt=0)
    connect_timeout_s: float = Field(30.0, gt=0)
    max_tensor_mb: float = Field(1024.0, gt=0)


class MeshTrainConfig(_Strict):
    job: JobConfig = JobConfig()
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    placement: PlacementConfig = PlacementConfig()
    workers: WorkersConfig = WorkersConfig()
    network: NetworkConfig = NetworkConfig()

    @model_validator(mode="after")
    def _check(self):
        if self.placement.stages:
            n = self.model_num_layers()
            if self.placement.stages[0].layers[0] != 0 or self.placement.stages[-1].layers[1] != n:
                raise ValueError(f"manual stages must cover layers [0, {n})")
        return self

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

"""Tensor identity and the TensorStore interface (V2 preparation, V1.5 scope).

V1.5 does **not** move tensors between memory tiers or workers: every
tensor lives on the accelerator of the stage that owns it for the whole run.
What V1.5 adds is the vocabulary V2 will need, so V2 does not require a
rewrite:

* stable tensor identities, e.g. ``model.layers.17.qkv.weight``
  (``.grad`` / ``.optim.exp_avg`` suffixes for gradients and optimizer
  state, ``act.s3.mb2.stage1.input`` for activations);
* ``TensorMeta``: id, owner, shape, dtype, bytes, role, tier;
* ``MemoryTier``: where a tensor can live (only LOCAL_ACCELERATOR / LOCAL_RAM
  are used in V1.5);
* ``TensorStore``: locate / put / get / evict / prefetch.

``LocalTensorStore`` implements it for the static V1.5 world: it records
what the local stage owns, answers ``locate`` and byte totals per role, and
refuses ``evict`` (that is V2 work: offload, paging, remote storage).
"""

from __future__ import annotations

import abc
import enum
import threading
from dataclasses import dataclass

import torch


class MemoryTier(enum.Enum):
    LOCAL_ACCELERATOR = "local_accelerator"   # V1.5: everything on a GPU/MPS stage lives here
    LOCAL_RAM = "local_ram"                   # V1.5: everything on a CPU stage lives here
    REMOTE_ACCELERATOR = "remote_accelerator"
    REMOTE_RAM = "remote_ram"
    LOCAL_NVME = "local_nvme"
    REMOTE_NVME = "remote_nvme"
    RECOMPUTE = "recompute"                   # dropped, recomputed from an earlier checkpoint


class TensorRole(enum.Enum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATION = "activation"
    ACTIVATION_GRADIENT = "activation_gradient"


@dataclass
class TensorMeta:
    tensor_id: str
    owner: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    role: TensorRole
    tier: MemoryTier

    def to_dict(self) -> dict:
        return {"tensor_id": self.tensor_id, "owner": self.owner, "shape": list(self.shape), "dtype": self.dtype,
                "bytes": self.bytes, "role": self.role.value, "tier": self.tier.value}


def parameter_id(layer_index: int, param_name: str) -> str:
    """Global, stable name of a parameter: ``model.layers.<global layer>.<name inside the layer>``."""
    return f"model.layers.{layer_index}.{param_name}"


def activation_id(step: int, microbatch: int, stage: int, kind: str = "input") -> str:
    return f"act.s{step}.mb{microbatch}.stage{stage}.{kind}"


class TensorStore(abc.ABC):
    """Where named tensors live. V2 implementations may move them between tiers."""

    @abc.abstractmethod
    def put(self, tensor_id: str, tensor: torch.Tensor, role: TensorRole, tier: MemoryTier | None = None) -> TensorMeta: ...

    @abc.abstractmethod
    def get(self, tensor_id: str) -> torch.Tensor: ...

    @abc.abstractmethod
    def locate(self, tensor_id: str) -> TensorMeta: ...

    @abc.abstractmethod
    def evict(self, tensor_id: str, to: MemoryTier) -> None: ...

    @abc.abstractmethod
    def prefetch(self, tensor_id: str) -> None: ...


class LocalTensorStore(TensorStore):
    """Tracks the tensors owned by one local pipeline stage (no movement in V1.5)."""

    def __init__(self, owner: str, device_type: str = "cpu"):
        self.owner = owner
        self.home = MemoryTier.LOCAL_RAM if device_type == "cpu" else MemoryTier.LOCAL_ACCELERATOR
        self._tensors: dict[str, torch.Tensor] = {}
        self._meta: dict[str, TensorMeta] = {}
        self._lock = threading.Lock()

    def put(self, tensor_id, tensor, role, tier=None):
        tier = tier or self.home
        if tier not in (MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM):
            raise NotImplementedError(f"{tier.value} storage is V2 work; V1.5 keeps tensors where they are")
        meta = TensorMeta(tensor_id, self.owner, tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""),
                          tensor.numel() * tensor.element_size(), role, tier)
        with self._lock:
            self._tensors[tensor_id] = tensor
            self._meta[tensor_id] = meta
        return meta

    def discard(self, tensor_id: str) -> None:
        with self._lock:
            self._tensors.pop(tensor_id, None)
            self._meta.pop(tensor_id, None)

    def get(self, tensor_id):
        try:
            return self._tensors[tensor_id]
        except KeyError:
            raise KeyError(f"tensor {tensor_id!r} is not owned by {self.owner}") from None

    def locate(self, tensor_id):
        try:
            return self._meta[tensor_id]
        except KeyError:
            raise KeyError(f"tensor {tensor_id!r} is not owned by {self.owner}") from None

    def evict(self, tensor_id, to):
        raise NotImplementedError("eviction/offload is V2 work (static ownership in V1.5)")

    def prefetch(self, tensor_id):
        self.locate(tensor_id)  # local tensors are always resident: nothing to do

    def ids(self, role: TensorRole | None = None) -> list[str]:
        with self._lock:
            return sorted(k for k, m in self._meta.items() if role is None or m.role == role)

    def bytes_by_role(self) -> dict[str, int]:
        out = {r.value: 0 for r in TensorRole}
        with self._lock:
            for m in self._meta.values():
                out[m.role.value] += m.bytes
        return out

    def manifest(self) -> list[dict]:
        with self._lock:
            return [self._meta[k].to_dict() for k in sorted(self._meta)]

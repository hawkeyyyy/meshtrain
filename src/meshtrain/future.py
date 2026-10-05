"""Interfaces reserved for post-V1 research. NOTHING here is implemented.

V1 keeps every tensor on the accelerator of the stage that owns it. These
abstract types document where later work (tensor offload, recompute,
compression, live migration) would plug in, so V1 code does not need to be
redesigned. Do not add behaviour here without an explicit request.
"""

from __future__ import annotations

import abc


# MemoryTier, TensorStore and the V1.5 LocalTensorStore live in runtime/tensor_store.py.
from meshtrain.runtime.tensor_store import LocalTensorStore, MemoryTier, TensorRole, TensorStore  # noqa: E402,F401


class PlacementPolicy(abc.ABC):
    """Chooses a tier for each tensor given memory pressure and link costs."""

    @abc.abstractmethod
    def place(self, key: str, nbytes: int, access_pattern: str) -> MemoryTier: ...


class RecomputePolicy(abc.ABC):
    """Decides which activations to drop and recompute during backward."""

    @abc.abstractmethod
    def should_recompute(self, layer_index: int, saved_bytes: int) -> bool: ...


class CompressionPolicy(abc.ABC):
    """Lossy/lossless compression of tensors on a given link."""

    @abc.abstractmethod
    def encode(self, tensor, link: tuple[str, str]) -> tuple[bytes, dict]: ...

    @abc.abstractmethod
    def decode(self, payload: bytes, meta: dict): ...


class MigrationPolicy(abc.ABC):
    """Moves layers between workers between steps (V1 placement is static)."""

    @abc.abstractmethod
    def propose(self, step: int, metrics: dict) -> list[tuple[int, str, str]]: ...

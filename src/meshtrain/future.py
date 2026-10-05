"""Interfaces reserved for post-V1 research. NOTHING here is implemented.

V1 keeps every tensor on the accelerator of the stage that owns it. These
abstract types document where later work (tensor offload, recompute,
compression, live migration) would plug in, so V1 code does not need to be
redesigned. Do not add behaviour here without an explicit request.
"""

from __future__ import annotations

import abc
import enum


class MemoryTier(enum.Enum):
    LOCAL_ACCELERATOR = "local_accelerator"   # the only tier used by V1
    LOCAL_RAM = "local_ram"
    REMOTE_ACCELERATOR = "remote_accelerator"
    REMOTE_RAM = "remote_ram"
    LOCAL_NVME = "local_nvme"
    REMOTE_NVME = "remote_nvme"
    RECOMPUTE = "recompute"                   # drop and recompute from an earlier checkpoint


class TensorStore(abc.ABC):
    """Where a named tensor (weight, optimizer state, saved activation) lives."""

    @abc.abstractmethod
    def put(self, key: str, tensor, tier: MemoryTier) -> None: ...

    @abc.abstractmethod
    def get(self, key: str, device): ...

    @abc.abstractmethod
    def tier_of(self, key: str) -> MemoryTier: ...


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

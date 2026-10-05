"""Model specification shared by the runtime, planner and experiments.

A MeshTrain model is an *ordered list of layers* plus a loss and a synthetic
data generator. Every layer is initialised from its own seed, so any worker
can build exactly its contiguous slice ``layers[start:end]`` and get the
same parameters the single-process reference model would have.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch import nn


def layer_seed(seed: int, index: int) -> int:
    return (seed * 1_000_003 + index * 7919) % (2**31 - 1)


class ModelSpec(abc.ABC):
    name: str = "model"

    def __init__(self, seed: int = 0, dtype: torch.dtype = torch.float32):
        self.seed = seed
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def num_layers(self) -> int: ...

    @abc.abstractmethod
    def _make_layer(self, index: int) -> nn.Module: ...

    @abc.abstractmethod
    def layer_name(self, index: int) -> str: ...

    @abc.abstractmethod
    def loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def make_batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic synthetic (inputs, targets) on CPU for a given step."""

    def build_layer(self, index: int) -> nn.Module:
        if not 0 <= index < self.num_layers:
            raise IndexError(index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(layer_seed(self.seed, index))
            layer = self._make_layer(index)
        return layer.to(self.dtype)

    def build_stage(self, start: int, end: int) -> nn.Sequential:
        """Contiguous layer range [start, end)."""
        if not 0 <= start < end <= self.num_layers:
            raise ValueError(f"invalid layer range [{start}, {end}) for {self.num_layers} layers")
        return nn.Sequential(*[self.build_layer(i) for i in range(start, end)])

    def build_full(self) -> nn.Sequential:
        return self.build_stage(0, self.num_layers)

    def describe(self) -> dict:
        return {"type": self.name, "num_layers": self.num_layers, "seed": self.seed,
                "parameters": self.parameter_count()}

    def parameter_count(self) -> int:
        with torch.device("meta"):
            return sum(p.numel() for i in range(self.num_layers) for p in self._make_layer(i).parameters())


@dataclass
class StageRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

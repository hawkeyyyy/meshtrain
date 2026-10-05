"""Small deterministic MLP used to prove distributed-autograd correctness.

Default architecture (7 partitionable layers):

    Linear(128, 512), GELU, Linear(512, 512), GELU, Linear(512, 256), GELU, Linear(256, 10)

Synthetic task: 10-way classification where the label is the argmax of a
fixed random linear "teacher" applied to Gaussian inputs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from meshtrain.models.base import ModelSpec


class MLPSpec(ModelSpec):
    name = "mlp"

    def __init__(self, sizes: list[int] | None = None, seed: int = 0, dtype: torch.dtype = torch.float32):
        super().__init__(seed, dtype)
        self.sizes = list(sizes or [128, 512, 512, 256, 10])
        if len(self.sizes) < 2:
            raise ValueError("MLP needs at least input and output size")
        # Alternating Linear / GELU, no activation after the last Linear.
        self._layers: list[tuple[str, int]] = []
        for i in range(len(self.sizes) - 1):
            self._layers.append(("linear", i))
            if i < len(self.sizes) - 2:
                self._layers.append(("gelu", i))
        g = torch.Generator().manual_seed(seed + 12345)
        self._teacher = torch.randn(self.sizes[0], self.sizes[-1], generator=g)

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def layer_name(self, index: int) -> str:
        kind, i = self._layers[index]
        return f"{kind}{i}"

    def _make_layer(self, index: int) -> nn.Module:
        kind, i = self._layers[index]
        if kind == "linear":
            return nn.Linear(self.sizes[i], self.sizes[i + 1])
        return nn.GELU()

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def make_batch(self, step: int, batch_size: int):
        g = torch.Generator().manual_seed(self.seed * 100_003 + step)
        x = torch.randn(batch_size, self.sizes[0], generator=g)
        y = (x @ self._teacher).argmax(dim=-1)
        return x.to(self.dtype), y

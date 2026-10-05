"""Model registry."""

from __future__ import annotations

import torch

from meshtrain.models.base import ModelSpec
from meshtrain.models.mlp_test import MLPSpec
from meshtrain.models.tiny_transformer import TinyTransformerSpec

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_model_spec(cfg: dict, seed: int = 0) -> ModelSpec:
    """Build a ModelSpec from the ``model:`` section of a config."""
    cfg = dict(cfg)
    kind = cfg.pop("type")
    dtype = _DTYPES[cfg.pop("dtype", "float32")]
    if kind == "mlp":
        return MLPSpec(sizes=cfg.get("sizes"), seed=seed, dtype=dtype)
    if kind == "tiny_transformer":
        return TinyTransformerSpec(
            layers=cfg.get("layers", 6),
            hidden_size=cfg.get("hidden_size", 256),
            heads=cfg.get("heads", 4),
            vocab_size=cfg.get("vocab_size", 256),
            seq_len=cfg.get("seq_len", 64),
            seed=seed,
            dtype=dtype,
        )
    raise ValueError(f"unknown model type {kind!r}")


__all__ = ["ModelSpec", "MLPSpec", "TinyTransformerSpec", "build_model_spec"]

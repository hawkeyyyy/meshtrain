"""Tiny decoder-only Transformer expressed as a layer list.

Layers: [Embedding] + n_layers x [TransformerBlock] + [LMHead].

Synthetic next-token task: each sequence is an arithmetic progression
``tok[t] = (a + b*t) mod vocab`` with random ``a`` and ``b`` per sequence,
so the next token is predictable from the previous two tokens.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from meshtrain.models.base import ModelSpec


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.tok(tokens) + self.pos(positions)[None]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.out(self.ln(x))


class TinyTransformerSpec(ModelSpec):
    name = "tiny_transformer"
    required_ops = ("embedding", "linear", "layer_norm", "gelu", "sdpa", "cross_entropy")

    def __init__(self, layers: int = 6, hidden_size: int = 256, heads: int = 4, vocab_size: int = 256,
                 seq_len: int = 64, seed: int = 0, dtype: torch.dtype = torch.float32):
        super().__init__(seed, dtype)
        self.n_blocks = layers
        self.d_model = hidden_size
        self.n_heads = heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    @property
    def num_layers(self) -> int:
        return self.n_blocks + 2

    def layer_name(self, index: int) -> str:
        if index == 0:
            return "embedding"
        if index == self.num_layers - 1:
            return "lm_head"
        return f"block{index - 1}"

    def _make_layer(self, index: int) -> nn.Module:
        if index == 0:
            return Embedding(self.vocab_size, self.d_model, self.seq_len)
        if index == self.num_layers - 1:
            return LMHead(self.d_model, self.vocab_size)
        return TransformerBlock(self.d_model, self.n_heads)

    def loss_fn(self, output, target):
        return F.cross_entropy(output.reshape(-1, output.shape[-1]).float(), target.reshape(-1))

    def make_batch(self, step: int, batch_size: int):
        g = torch.Generator().manual_seed(self.seed * 100_003 + step)
        a = torch.randint(0, self.vocab_size, (batch_size, 1), generator=g)
        b = torch.randint(1, 8, (batch_size, 1), generator=g)
        t = torch.arange(self.seq_len + 1)[None]
        seq = (a + b * t) % self.vocab_size
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

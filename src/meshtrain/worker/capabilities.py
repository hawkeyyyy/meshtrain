"""Backend capability probing: ``supports(op, dtype)`` answered by running it.

Each probe runs a tiny forward (and backward where relevant) of one operator
on the device. Results are cached per (backend, op, dtype). This is not
exhaustive operator coverage -- it covers the operators MeshTrain's models
use, so the planner can refuse to place a stage on a backend that cannot
run it instead of failing mid-training.

Models declare what they need via ``ModelSpec.required_ops``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}


def _grad(x: torch.Tensor) -> torch.Tensor:
    return x.detach().requires_grad_(True)


def _p_matmul(dev, dt):
    a, b = _grad(torch.randn(8, 8, device=dev, dtype=dt)), torch.randn(8, 8, device=dev, dtype=dt)
    (a @ b).sum().backward()


def _p_linear(dev, dt):
    lin = torch.nn.Linear(8, 8).to(device=dev, dtype=dt)
    lin(torch.randn(2, 8, device=dev, dtype=dt)).sum().backward()


def _p_gelu(dev, dt):
    F.gelu(_grad(torch.randn(16, device=dev, dtype=dt))).sum().backward()


def _p_layer_norm(dev, dt):
    ln = torch.nn.LayerNorm(8).to(device=dev, dtype=dt)
    ln(torch.randn(2, 8, device=dev, dtype=dt)).sum().backward()


def _p_softmax(dev, dt):
    torch.softmax(_grad(torch.randn(4, 8, device=dev, dtype=dt)), -1).sum().backward()


def _p_sdpa(dev, dt):
    q = _grad(torch.randn(1, 2, 4, 8, device=dev, dtype=dt))
    F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()


def _p_embedding(dev, dt):
    emb = torch.nn.Embedding(16, 8).to(device=dev, dtype=dt)
    emb(torch.tensor([[1, 2, 3]], device=dev)).sum().backward()


def _p_cross_entropy(dev, dt):
    logits = _grad(torch.randn(4, 10, device=dev, dtype=dt))
    F.cross_entropy(logits.float(), torch.tensor([1, 2, 3, 4], device=dev)).backward()


def _p_adamw(dev, dt):
    p = torch.nn.Parameter(torch.randn(8, device=dev, dtype=dt))
    p.grad = torch.randn(8, device=dev, dtype=dt)
    torch.optim.AdamW([p], lr=1e-3).step()


PROBES = {
    "matmul": _p_matmul, "linear": _p_linear, "gelu": _p_gelu, "layer_norm": _p_layer_norm,
    "softmax": _p_softmax, "sdpa": _p_sdpa, "embedding": _p_embedding, "cross_entropy": _p_cross_entropy,
    "adamw": _p_adamw,
}


class CapabilityProber:
    def __init__(self, device: torch.device, dtype_ok):
        self.device = device
        self.dtype_ok = dtype_ok  # adapter's static dtype rule (e.g. MPS: no float64)
        self._cache: dict[tuple[str, str], bool] = {}
        self.errors: dict[str, str] = {}

    def supports(self, op: str, dtype: str = "float32") -> bool:
        key = (op, dtype)
        if key in self._cache:
            return self._cache[key]
        torch_dtype = DTYPES.get(dtype)
        if torch_dtype is None or not self.dtype_ok(torch_dtype):
            ok = False
        elif op == "dtype":
            ok = self._run(lambda d, t: torch.ones(4, device=d, dtype=t).sum().item(), op, dtype)
        elif op in PROBES:
            ok = self._run(PROBES[op], op, dtype)
        else:
            ok = True  # unknown ops: let torch report the real error
        self._cache[key] = ok
        return ok

    def _run(self, fn, op, dtype) -> bool:
        try:
            with torch.random.fork_rng(devices=[]):
                fn(self.device, DTYPES[dtype])
            return True
        except Exception as exc:  # unsupported kernel / dtype on this backend
            self.errors[f"{op}:{dtype}"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            return False

    def report(self, ops=tuple(PROBES), dtypes=("float32", "float16", "bfloat16")) -> dict:
        return {
            "supports_fp32": self.supports("dtype", "float32"),
            "supports_fp16": self.supports("dtype", "float16"),
            "supports_bf16": self.supports("dtype", "bfloat16"),
            "supports_fp64": self.supports("dtype", "float64"),
            "op_capabilities": {dt: sorted(op for op in ops if self.supports(op, dt)) for dt in dtypes},
            "probe_errors": dict(self.errors),
        }

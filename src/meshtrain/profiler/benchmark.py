"""Compute micro-benchmarks for a worker's device.

``compute_score`` is *not* hard-coded: each worker measures fp32 matmul
throughput and MLP forward/backward time on its device; the coordinator
normalises scores relative to the fastest worker.
"""

from __future__ import annotations

import statistics
import time

import torch
from torch import nn

from meshtrain.worker.device import DeviceAdapter


def _timeit(fn, adapter: DeviceAdapter, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    adapter.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        adapter.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def benchmark_device(adapter: DeviceAdapter, *, matmul_size: int = 1024, iters: int = 10,
                     mlp_batch: int = 64, mlp_hidden: int = 1024, quick: bool = False) -> dict:
    if quick:
        matmul_size, iters, mlp_hidden = 512, 3, 512
    dev = adapter.device
    g = torch.Generator().manual_seed(0)
    a = adapter.move_tensor(torch.randn(matmul_size, matmul_size, generator=g))
    b = adapter.move_tensor(torch.randn(matmul_size, matmul_size, generator=g))
    mm_s = _timeit(lambda: a @ b, adapter, warmup=2, iters=iters)
    gflops = 2 * matmul_size**3 / mm_s / 1e9

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, mlp_hidden),
                            nn.GELU(), nn.Linear(mlp_hidden, mlp_hidden))
    mlp = adapter.move_module(mlp)
    x = adapter.move_tensor(torch.randn(mlp_batch, mlp_hidden, generator=g))

    def fwd():
        with torch.no_grad():
            mlp(x)

    def fwd_bwd():
        mlp(x).sum().backward()

    fwd_s = _timeit(fwd, adapter, warmup=2, iters=iters)
    fwd_bwd_s = _timeit(fwd_bwd, adapter, warmup=2, iters=iters)

    host = torch.randn(4 * 1024 * 1024 // 4)  # 4 MB
    def h2d():
        adapter.move_tensor(host)
    h2d_s = _timeit(h2d, adapter, warmup=1, iters=max(3, iters // 2))
    del a, b, x
    return {
        "backend": adapter.backend,
        "device": str(dev),
        "device_name": adapter.name(),
        "matmul_size": matmul_size,
        "matmul_s": mm_s,
        "matmul_gflops": gflops,
        "mlp_forward_s": fwd_s,
        "mlp_backward_s": max(fwd_bwd_s - fwd_s, 0.0),
        "mlp_fwd_bwd_s": fwd_bwd_s,
        "host_to_device_GBps": host.numel() * 4 / h2d_s / 1e9 if h2d_s > 0 else float("inf"),
        # Effective FLOP/s used by the planner's cost model (matmul-dominated).
        "measured_flops": gflops * 1e9,
    }


def normalise_scores(benchmarks: dict[str, dict]) -> dict[str, float]:
    """compute_score[w] = measured_flops[w] / max_w measured_flops."""
    if not benchmarks:
        return {}
    best = max(b["measured_flops"] for b in benchmarks.values())
    return {w: b["measured_flops"] / best for w, b in benchmarks.items()}

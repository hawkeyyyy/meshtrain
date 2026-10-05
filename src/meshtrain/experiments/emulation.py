"""Emulated heterogeneous workers on a single CPU host (for experiments only).

When no physical heterogeneous cluster is available, experiments can run
against *emulated* workers: separate CPU processes with different torch
thread counts (real, measured compute differences) and declared memory
budgets (enforced by the planner's memory accounting, then checked against
the tensor bytes each stage actually held).

Everything produced this way is labelled "emulated" in docs/v1-results.md.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

from meshtrain.planner.partition import WorkerProfile

MB = 1024**2


@dataclass
class EmulatedWorker:
    name: str
    threads: int
    memory_budget_bytes: int  # emulated accelerator memory (total, before headroom)
    label: str = ""


def _bench(threads: int) -> dict:
    import torch

    from meshtrain.profiler.benchmark import benchmark_device
    from meshtrain.worker.device import CPUDeviceAdapter

    torch.set_num_threads(threads)
    return benchmark_device(CPUDeviceAdapter(), matmul_size=768, iters=8)


def profile_emulated(workers: list[EmulatedWorker]) -> list[WorkerProfile]:
    """Benchmark each emulated worker in its own process with its thread count."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:  # sequential: measurements must not compete for cores
        results = [pool.apply(_bench, (w.threads,)) for w in workers]
    best = max(r["measured_flops"] for r in results)
    return [WorkerProfile(w.name, w.name, "cpu", w.memory_budget_bytes, r["measured_flops"],
                          compute_score=r["measured_flops"] / best) for w, r in zip(workers, results)]

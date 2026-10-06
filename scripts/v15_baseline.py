"""Single-device V1.5 baseline: throughput, peak memory and static capacity.

Uses only V1.5 APIs (Stage + run_stage, one stage, no network), so it runs on
the V1.5 tag and on V2 (where it measures ``memory.strategy: static``).

    python scripts/v15_baseline.py throughput --device cuda
    python scripts/v15_baseline.py capacity --device cuda --cap-gb 8
    python scripts/v15_baseline.py capacity --device cuda --cap-gb 2

``--cap-gb`` hard-caps the CUDA caching allocator (torch.cuda.set_per_process_memory_fraction),
which also stops the Windows driver from silently spilling into shared system memory.
Each capacity trial runs in a fresh process. Output: JSON lines on stdout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

THROUGHPUT_MODELS = [
    {"type": "tiny_transformer", "layers": 12, "hidden_size": 512, "heads": 8, "vocab_size": 1024, "seq_len": 128},
    {"type": "tiny_transformer", "layers": 24, "hidden_size": 1024, "heads": 16, "vocab_size": 1024, "seq_len": 128},
]
CAPACITY_SIZES = [(4, 1280), (8, 1280), (12, 1280), (16, 1280), (20, 1280), (24, 1280), (28, 1280), (32, 1280),
                  (40, 1280), (48, 1280)]


def _trial(model_cfg: dict, device: str, steps: int, batch: int, micro: int, cap_gb: float | None,
           optimizer: str = "adamw") -> dict:
    import torch

    from meshtrain.models import build_model_spec
    from meshtrain.runtime.pipeline import PipelineSettings, run_stage
    from meshtrain.runtime.stage import Stage
    from meshtrain.worker.device import select_device

    dev = select_device(device)
    if cap_gb is not None and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 1024**3 / dev.memory_total()), dev.index)
    spec = build_model_spec(model_cfg, seed=0)
    params = spec.parameter_count()
    out = {"model": model_cfg, "parameters": params, "device": device, "device_name": dev.name(),
           "cap_gb": cap_gb, "batch": batch, "microbatches": batch // micro, "steps": steps}
    try:
        stage = Stage(spec.build_full(), stage_index=0, num_stages=1, device=dev, optimizer=optimizer, lr=3e-4,
                      loss_fn=spec.loss_fn, name="baseline")
        settings = PipelineSettings("baseline", steps=steps, batch_size=batch, num_microbatches=batch // micro,
                                    log_every=10**6, schedule="1f1b", trace=False)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(dev.index)
        t0 = time.perf_counter()
        res = run_stage(stage, spec, settings, upstream=None, downstream=None, worker="baseline")
        wall = time.perf_counter() - t0
        steady = [m["step_s"] for m in res.step_metrics[1:]] or [res.step_metrics[0]["step_s"]]
        out.update(ok=True, loss_first=res.losses[0], loss_last=res.losses[-1], wall_s=wall,
                   step_s=sum(steady) / len(steady), samples_per_s=batch / (sum(steady) / len(steady)),
                   peak_allocated=int(torch.cuda.max_memory_allocated(dev.index)) if device == "cuda" else None)
    except Exception as exc:  # OOM is the expected failure mode
        out.update(ok=False, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    return out


def _child(q, *a):
    try:
        q.put(_trial(*a))
    except BaseException as exc:
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def run_isolated(*a, timeout: float = 900.0) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, *a))
    p.start()
    try:
        return q.get(timeout=timeout)
    except Exception:
        return {"ok": False, "error": "timeout or crash", "model": a[0]}
    finally:
        p.join(10)
        if p.is_alive():
            p.terminate()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["throughput", "capacity"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap-gb", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    if a.what == "throughput":
        for m in THROUGHPUT_MODELS:
            print(json.dumps(run_isolated(m, a.device, a.steps or 12, 16, 4, a.cap_gb)), flush=True)
    else:
        for blocks, hidden in CAPACITY_SIZES:
            m = {"type": "tiny_transformer", "layers": blocks, "hidden_size": hidden, "heads": 8,
                 "vocab_size": 1024, "seq_len": 64}
            r = run_isolated(m, a.device, a.steps or 3, 8, 2, a.cap_gb)
            print(json.dumps(r), flush=True)
            if not r.get("ok"):
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run a whole pipeline on one machine, one OS process per stage.

This is the Milestone-1 harness (and the Experiment-1 driver): stages are
real separate processes that exchange serialized TensorPackets, either over
multiprocessing pipes (``transport="pipe"``) or over loopback TCP
(``transport="tcp"``), exactly like remote workers would.

Also provides ``reference_step`` -- the single-process PyTorch baseline used
for gradient-equivalence checks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass

import torch

from meshtrain.models import build_model_spec
from meshtrain.runtime.pipeline import PipelineSettings, StageResult, run_stage
from meshtrain.runtime.stage import Stage, build_optimizer
from meshtrain.telemetry import EventLogger


@dataclass
class LocalStage:
    layers: tuple[int, int]
    device: str = "cpu"
    threads: int | None = None  # per-process torch threads (emulates slower/faster CPU workers)


def _stage_process(index, stages, model_cfg, seed, settings, optimizer, lr, transport, links,
                   results, capture_params, torch_threads, link_emulation=None, residency=None,
                   accelerator_budget=None):
    os.environ.setdefault("MESHTRAIN_QUIET", "1")
    threads = stages[index].threads or torch_threads
    if threads:
        torch.set_num_threads(threads)
    from meshtrain.worker.device import select_device

    up = down = None
    worker = f"local{index}"
    try:
        spec = build_model_spec(model_cfg, seed=seed)
        st = stages[index]
        if transport == "pipe":
            from meshtrain.networking.transport import PipeTransport

            up = PipeTransport(links[index - 1][1]) if index > 0 else None
            down = PipeTransport(links[index][0]) if index < len(stages) - 1 else None
        else:
            from meshtrain.networking.tcp import TCPListener, connect

            listener = None
            if index > 0:
                listener = TCPListener("127.0.0.1", 0)
                results.put(("port", index, listener.port))
            if index < len(stages) - 1:
                port = links.get_port(index + 1)
                down = connect("127.0.0.1", port, timeout=settings.timeout_s,
                               hello={"job_id": settings.job_id, "stage": index})
            if listener is not None:
                up, _hello = listener.accept(timeout=settings.timeout_s)
                listener.close()
        if link_emulation is not None:
            from meshtrain.networking.emulation import EmulatedLink

            bw, lat = link_emulation
            up = EmulatedLink(up, bw, lat) if up is not None else None
            down = EmulatedLink(down, bw, lat) if down is not None else None
        adapter = select_device(st.device)
        policy = residency
        if residency is not None:
            from meshtrain.planner.residency import resolve_stage_policy

            M = settings.num_microbatches
            policy, _plan = resolve_stage_policy(residency, spec, *st.layers, microbatch_size=settings.batch_size // M,
                                                 budget=accelerator_budget, optimizer=optimizer,
                                                 backend=adapter.backend, num_microbatches=M,
                                                 schedule=settings.schedule, stage_index=index,
                                                 num_stages=len(stages))
        stage = Stage(spec.build_stage(*st.layers), stage_index=index, num_stages=len(stages),
                      device=adapter, optimizer=optimizer, lr=lr, loss_fn=spec.loss_fn,
                      name=worker, layer_offset=st.layers[0], residency=policy,
                      accelerator_budget=accelerator_budget)
        logger = EventLogger(worker, settings.job_id)
        res = run_stage(stage, spec, settings, upstream=up, downstream=down, worker=worker,
                        logger=logger, capture_params=capture_params)
        results.put(("result", index, _pack_result(res)))
    except Exception as exc:
        results.put(("error", index, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        for link in (up, down):
            if link is not None:
                link.close()


_TENSOR_FIELDS = ("gradients", "initial_params", "final_params")


def _pack_result(res: StageResult) -> StageResult:
    """Replace tensors with raw bytes so results survive the child's exit
    (torch's shared-memory pickling needs the producer process alive)."""
    from meshtrain.runtime.serialization import tensor_to_bytes

    for f in _TENSOR_FIELDS:
        d = getattr(res, f)
        if d is not None:
            setattr(res, f, {k: tensor_to_bytes(v) for k, v in d.items()})
    return res


def _unpack_result(res: StageResult) -> StageResult:
    from meshtrain.runtime.serialization import bytes_to_tensor

    for f in _TENSOR_FIELDS:
        d = getattr(res, f)
        if d is not None:
            setattr(res, f, {k: bytes_to_tensor(*v) for k, v in d.items()})
    return res


class _PortBoard:
    """Lets the TCP-mode child for stage i learn the port of stage i+1."""

    def __init__(self, manager_dict):
        self.d = manager_dict

    def get_port(self, index, timeout=30.0):
        import time

        t0 = time.time()
        while index not in self.d:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"stage {index} never published its port")
            time.sleep(0.01)
        return self.d[index]


def run_local_pipeline(
    model_cfg: dict,
    stages: list[LocalStage],
    settings: PipelineSettings,
    *,
    seed: int = 0,
    optimizer: str = "sgd",
    lr: float = 0.01,
    transport: str = "pipe",
    capture_params: bool = False,
    torch_threads: int | None = 1,
    timeout_s: float = 600.0,
    link_emulation: tuple[float, float] | None = None,
    residency=None,
    accelerator_budget: int | None = None,
) -> list[StageResult]:
    """``link_emulation=(bandwidth_Bps, latency_s)`` throttles every link (experiments only).
    ``residency`` (runtime.offload.ResidencyPolicy) / ``accelerator_budget`` apply to every stage (V2)."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    manager = None
    if transport == "pipe":
        links = [ctx.Pipe(duplex=True) for _ in range(len(stages) - 1)]
    elif transport == "tcp":
        manager = ctx.Manager()
        links = _PortBoard(manager.dict())
    else:
        raise ValueError(f"unknown local transport {transport!r}")
    procs = [
        ctx.Process(target=_stage_process, name=f"meshtrain-stage{i}",
                    args=(i, stages, model_cfg, seed, settings, optimizer, lr, transport, links, results,
                          capture_params, torch_threads, link_emulation, residency, accelerator_budget))
        for i in range(len(stages))
    ]
    for p in procs:
        p.start()
    out: dict[int, StageResult] = {}
    errors: dict[int, str] = {}
    try:
        while len(out) + len(errors) < len(stages):
            try:
                kind, index, payload = results.get(timeout=timeout_s)
            except queue.Empty:
                raise TimeoutError(f"local pipeline produced no result within {timeout_s}s") from None
            if kind == "port":
                links.d[index] = payload
            elif kind == "result":
                out[index] = _unpack_result(payload)
            else:
                errors[index] = payload
                # A failed stage makes its neighbours fail via ERROR packets or
                # timeouts; keep collecting so we report every stage.
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        if manager is not None:
            manager.shutdown()
    if errors:
        msg = "\n".join(f"stage {i}: {e}" for i, e in sorted(errors.items()))
        raise RuntimeError(f"local pipeline failed:\n{msg}")
    return [out[i] for i in range(len(stages))]


def reference_step(model_cfg: dict, settings: PipelineSettings, *, seed: int = 0, optimizer: str = "sgd",
                   lr: float = 0.01, steps: int = 1, device: str = "cpu"):
    """Single-process baseline: same layers, same data, gradient accumulation
    over the same microbatches. Returns (losses, grads of last step keyed
    ``layer_index.param``, final params)."""
    from meshtrain.worker.device import select_device

    spec = build_model_spec(model_cfg, seed=seed)
    adapter = select_device(device)
    model = adapter.move_module(spec.build_full())
    opt = build_optimizer(optimizer, model.parameters(), lr)
    M = settings.num_microbatches
    losses, grads = [], {}
    for local_step in range(steps):
        step = settings.step_offset + local_step
        x, y = spec.make_batch(step, settings.batch_size)
        total = 0.0
        for xm, ym in zip(x.chunk(M), y.chunk(M)):
            loss = spec.loss_fn(model(adapter.move_tensor(xm)), adapter.move_tensor(ym))
            (loss / M).backward()
            total += loss.item()
        losses.append(total / M)
        grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
        opt.step()
        opt.zero_grad(set_to_none=True)
    params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    return losses, grads, params


def global_param_names(stages: list[LocalStage], local: dict[str, torch.Tensor], stage_index: int) -> dict[str, torch.Tensor]:
    """Map stage-local names ('0.weight') to full-model names ('3.weight')."""
    start = stages[stage_index].layers[0]
    out = {}
    for name, t in local.items():
        idx, rest = name.split(".", 1)
        out[f"{int(idx) + start}.{rest}"] = t
    return out

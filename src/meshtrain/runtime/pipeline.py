"""Synchronous GPipe-style pipeline execution for one stage.

Every stage runs ``run_stage`` with links to its neighbours:

* stage 0 owns the data: it splits each batch into M microbatches, runs all
  M forwards (sending activations + targets downstream), then waits for M
  gradients and backpropagates each into the saved graph of that microbatch.
* middle stages are event driven (forward on activation, backward on
  gradient, relay targets).
* the last stage pairs each activation with its target, computes the loss
  (scaled by 1/M) and backpropagates immediately.

After its M-th backward a stage runs its optimizer step. Because each stage
processes its inbox on one thread, step s+1 cannot start before step s's
update: training is synchronous and matches single-process gradient
accumulation.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from meshtrain.models.base import ModelSpec
from meshtrain.networking.transport import Transport, TransportClosed, TransportTimeout
from meshtrain.runtime.distributed_autograd import BoundaryError, MicrobatchContext
from meshtrain.runtime.serialization import TensorLifecycle, packet_to_tensor
from meshtrain.runtime.stage import Stage
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket
from meshtrain.telemetry import EventLogger

UPSTREAM = "upstream"
DOWNSTREAM = "downstream"


class PipelineError(RuntimeError):
    pass


class InboxTimeout(PipelineError):
    pass


@dataclass
class PipelineSettings:
    job_id: str
    steps: int
    batch_size: int
    num_microbatches: int = 1
    timeout_s: float = 60.0
    log_every: int = 10
    capture_gradients_at_step: int | None = None  # record grads before optimizer step
    step_offset: int = 0
    log_microbatch_events: bool = False


@dataclass
class StageResult:
    stage_index: int
    worker: str
    losses: list[float] = field(default_factory=list)  # stage 0 only: per-step mean loss
    step_metrics: list[dict] = field(default_factory=list)
    gradients: dict[str, torch.Tensor] | None = None
    initial_params: dict[str, torch.Tensor] | None = None
    final_params: dict[str, torch.Tensor] | None = None
    error: str | None = None


class Inbox:
    """Merges packets from several transports into one queue (one reader thread each)."""

    def __init__(self, links: dict[str, Transport], poll_s: float = 0.2):
        self.queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads = []
        for name, link in links.items():
            t = threading.Thread(target=self._reader, args=(name, link, poll_s), daemon=True,
                                 name=f"inbox-{name}")
            t.start()
            self._threads.append(t)

    def _reader(self, name: str, link: Transport, poll_s: float) -> None:
        while not self._stop.is_set():
            try:
                t0 = time.perf_counter()
                packet = link.recv_packet(timeout=poll_s)
                self.queue.put((name, packet, time.perf_counter() - t0, None))
            except TransportTimeout:
                continue
            except Exception as exc:  # closed peer, malformed frame, ...
                if not self._stop.is_set():
                    self.queue.put((name, None, 0.0, exc))
                return

    def get(self, timeout: float) -> tuple[str, TensorPacket, float]:
        try:
            name, packet, recv_s, exc = self.queue.get(timeout=timeout)
        except queue.Empty:
            raise InboxTimeout(f"timed out after {timeout}s waiting for a peer packet") from None
        if exc is not None:
            if isinstance(exc, TransportClosed):
                raise PipelineError(f"{name} peer disconnected: {exc}") from exc
            raise PipelineError(f"{name} link failed: {exc}") from exc
        if packet.message_type == MessageType.ERROR:
            raise PipelineError(f"{name} peer reported error: {packet.meta.get('error')}")
        return name, packet, recv_s

    def stop(self) -> None:
        self._stop.set()


class _StepStats:
    def __init__(self):
        self.forward_s = 0.0
        self.backward_s = 0.0
        self.comm_s = 0.0  # staging + (de)serialization + socket time
        self.idle_s = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.lifecycle: dict[str, float] = {}
        self.peak_saved_bytes = 0

    def add_lifecycle(self, lc: TensorLifecycle) -> None:
        for k, v in lc.timings.items():
            self.lifecycle[k] = self.lifecycle.get(k, 0.0) + v
            self.comm_s += v


def run_stage(
    stage: Stage,
    spec: ModelSpec,
    settings: PipelineSettings,
    *,
    upstream: Transport | None,
    downstream: Transport | None,
    worker: str = "local",
    logger: EventLogger | None = None,
    metrics_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    capture_params: bool = False,
) -> StageResult:
    logger = logger or EventLogger(worker, settings.job_id)
    if (upstream is None) != stage.is_first or (downstream is None) != stage.is_last:
        raise PipelineError("stage links do not match stage position")
    M = settings.num_microbatches
    if settings.batch_size % M:
        raise PipelineError("batch_size must be divisible by num_microbatches")
    links = {k: v for k, v in ((UPSTREAM, upstream), (DOWNSTREAM, downstream)) if v is not None}
    inbox = Inbox(links) if links else None
    result = StageResult(stage.stage_index, worker)
    if capture_params:
        result.initial_params = stage.named_parameters_cpu()
    dev = stage.device

    def send(link: Transport, tensor: torch.Tensor, mtype: MessageType, step: int, mb: int,
             stats: _StepStats, meta: dict | None = None) -> None:
        lc = TensorLifecycle()
        before = link.bytes_sent
        link.send_tensor(tensor, mtype, device=dev, lifecycle=lc, job_id=settings.job_id, step_id=step,
                         microbatch_id=mb, source_worker=worker, meta=meta or {})
        stats.add_lifecycle(lc)
        stats.bytes_sent += link.bytes_sent - before
        if settings.log_microbatch_events:
            direction = "→ downstream" if link is downstream else "→ upstream"
            logger.log("TRANSFER", step, mb, type=mtype.name, direction=direction, bytes=lc.nbytes,
                       time_s=sum(lc.timings.values()))

    def receive(stats: _StepStats) -> tuple[str, TensorPacket]:
        t0 = time.perf_counter()
        while True:
            if stop_event is not None and stop_event.is_set():
                raise PipelineError("job stopped")
            try:
                name, packet, _ = inbox.get(timeout=min(settings.timeout_s, 1.0))
                break
            except InboxTimeout:
                if time.perf_counter() - t0 < settings.timeout_s:
                    continue
                raise InboxTimeout(f"no packet from any peer within network timeout {settings.timeout_s}s") from None
        stats.idle_s += time.perf_counter() - t0
        stats.bytes_received += len(packet.payload)
        if packet.job_id != settings.job_id:
            raise BoundaryError(f"packet for job {packet.job_id!r} received in job {settings.job_id!r}")
        return name, packet

    def to_tensor(packet: TensorPacket, stats: _StepStats) -> torch.Tensor:
        lc = TensorLifecycle()
        t = packet_to_tensor(packet, device=dev, lifecycle=lc)
        stats.add_lifecycle(lc)
        return t

    try:
        for local_step in range(settings.steps):
            step = settings.step_offset + local_step
            stats = _StepStats()
            t_step = time.perf_counter()
            if stage.is_first:
                dev.reset_peak_memory()
            loss_sum = 0.0
            done = 0

            if stage.is_first:
                x, y = spec.make_batch(step, settings.batch_size)
                xs, ys = x.chunk(M), y.chunk(M)
                for mb in range(M):
                    ctx = MicrobatchContext(step, mb)
                    if stage.is_last:  # single-stage pipeline
                        loss, _ = stage.forward_loss(xs[mb], ys[mb], ctx, loss_scale=1.0 / M)
                        loss_sum += float(loss)
                        stats.forward_s += ctx.timings["forward"]
                        stats.backward_s += ctx.timings["backward"]
                        done += 1
                        continue
                    out = stage.forward(xs[mb], ctx)
                    stats.forward_s += ctx.timings["forward"]
                    stats.peak_saved_bytes = max(stats.peak_saved_bytes, stage.contexts.saved_bytes())
                    send(downstream, ys[mb], MessageType.TARGET, step, mb, stats)
                    send(downstream, out.detach(), MessageType.FORWARD_ACTIVATION, step, mb, stats)
                    if settings.log_microbatch_events:
                        logger.log("FORWARD_COMPLETE", step, mb, output=out.numel() * out.element_size(),
                                   compute_s=ctx.timings["forward"])

            pending_act: dict[int, torch.Tensor] = {}
            pending_tgt: dict[int, torch.Tensor] = {}
            while done < M:
                name, packet = receive(stats)
                mb = packet.microbatch_id
                if packet.step_id != step:
                    raise BoundaryError(f"packet for step {packet.step_id} while in step {step}")
                mtype = packet.message_type
                if mtype == MessageType.TARGET and name == UPSTREAM:
                    if stage.is_last:
                        pending_tgt[mb] = to_tensor(packet, stats)
                    else:  # relay unchanged
                        before = downstream.bytes_sent
                        packet.source_worker = worker
                        stats.comm_s += downstream.send_packet(packet)
                        stats.bytes_sent += downstream.bytes_sent - before
                elif mtype == MessageType.FORWARD_ACTIVATION and name == UPSTREAM:
                    act = to_tensor(packet, stats)
                    if stage.is_last:
                        pending_act[mb] = act
                    else:
                        ctx = MicrobatchContext(step, mb)
                        out = stage.forward(act, ctx)
                        stats.forward_s += ctx.timings["forward"]
                        stats.peak_saved_bytes = max(stats.peak_saved_bytes, stage.contexts.saved_bytes())
                        send(downstream, out.detach(), MessageType.FORWARD_ACTIVATION, step, mb, stats)
                        if settings.log_microbatch_events:
                            logger.log("FORWARD_COMPLETE", step, mb, output=out.numel() * out.element_size(),
                                       compute_s=ctx.timings["forward"])
                elif mtype == MessageType.BACKWARD_GRADIENT and name == DOWNSTREAM:
                    grad = to_tensor(packet, stats)
                    grad_in, ctx = stage.backward(grad, (step, mb))
                    stats.backward_s += ctx.timings["backward"]
                    loss_sum += float(packet.meta.get("loss", 0.0))
                    if not stage.is_first:
                        send(upstream, grad_in, MessageType.BACKWARD_GRADIENT, step, mb, stats, meta=packet.meta)
                    if settings.log_microbatch_events:
                        logger.log("BACKWARD_COMPLETE", step, mb, compute_s=ctx.timings["backward"])
                    done += 1
                else:
                    raise BoundaryError(f"unexpected {mtype.name} from {name}")

                if stage.is_last and mb in pending_act and mb in pending_tgt:
                    ctx = MicrobatchContext(step, mb)
                    loss, grad_in = stage.forward_loss(pending_act.pop(mb), pending_tgt.pop(mb), ctx,
                                                       loss_scale=1.0 / M)
                    stats.forward_s += ctx.timings["forward"]
                    stats.backward_s += ctx.timings["backward"]
                    loss_val = float(loss)
                    loss_sum += loss_val
                    send(upstream, grad_in, MessageType.BACKWARD_GRADIENT, step, mb, stats,
                         meta={"loss": loss_val})
                    if settings.log_microbatch_events:
                        logger.log("BACKWARD_COMPLETE", step, mb, loss=loss_val, compute_s=ctx.timings["backward"])
                    done += 1

            if settings.capture_gradients_at_step == step:
                result.gradients = stage.named_gradients()
            mem = stage.memory_report()  # gradients present, before zero_grad
            mem["saved_activations_peak"] = stats.peak_saved_bytes
            opt_s = stage.optimizer_step()
            stage.zero_grad()
            step_s = time.perf_counter() - t_step
            mean_loss = loss_sum / M
            dev_mem = dev.memory_stats()
            record = {
                "event": "STEP_COMPLETE", "job": settings.job_id, "worker": worker,
                "stage": stage.stage_index, "backend": dev.backend, "step": step,
                "step_s": step_s, "forward_s": stats.forward_s, "backward_s": stats.backward_s,
                "optimizer_s": opt_s, "comm_s": stats.comm_s, "idle_s": stats.idle_s,
                "bytes_sent": stats.bytes_sent, "bytes_received": stats.bytes_received,
                "lifecycle": stats.lifecycle,
                "utilization": (stats.forward_s + stats.backward_s + opt_s) / step_s if step_s else 0.0,
                "memory": {**mem, "device_allocated": dev_mem.get("allocated", 0),
                           "device_peak": dev_mem.get("peak_allocated", 0), "device_total": dev_mem.get("total", 0)},
            }
            if stage.is_first:
                record["loss"] = mean_loss
                record["samples_per_s"] = settings.batch_size / step_s if step_s else 0.0
                result.losses.append(mean_loss)
            result.step_metrics.append(record)
            if metrics_callback is not None:
                metrics_callback(record)
            if local_step % settings.log_every == 0 or local_step == settings.steps - 1:
                fields = {"step_s": step_s, "fwd_s": stats.forward_s, "bwd_s": stats.backward_s,
                          "comm_s": stats.comm_s, "idle_s": stats.idle_s, "sent_bytes": stats.bytes_sent}
                if stage.is_first:
                    fields = {"loss": mean_loss, **fields}
                logger.log("STEP_COMPLETE", step, None, **fields)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.log("ERROR", None, None, error=result.error)
        # Best effort: tell neighbours so they fail fast instead of timing out.
        for link in links.values():
            try:
                link.send_packet(TensorPacket(MessageType.ERROR, job_id=settings.job_id, source_worker=worker,
                                              meta={"error": f"{worker}: {result.error}"[:200]}))
            except Exception:
                pass
        raise
    finally:
        if inbox is not None:
            inbox.stop()
    if capture_params:
        result.final_params = stage.named_parameters_cpu()
    return result

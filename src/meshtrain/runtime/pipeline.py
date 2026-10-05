"""Schedule-driven pipeline execution for one stage (V1.5).

Each stage runs ``run_stage`` with links to its neighbours. Per global step::

    state machine  <- schedule (gpipe | 1f1b) for (stage, S, M)
    loop over actions in schedule order:
        wait until the action is ready  (WAIT_FORWARD / WAIT_BACKWARD spans)
            -- inbound events: activations, gradients, targets, send completions
        FORWARD:  stage.forward(...)  -> submit activation to the outbox
        BACKWARD: stage.backward(...) or loss_backward (last stage)
                  -> submit input-gradient upstream
    optimizer.step(); zero_grad()          (once per global step)

Threads per stage:

* compute thread  -- this function: schedule, forward/backward, optimizer,
                     host-to-device copies of received tensors.
* reader thread   -- one per link: receive + deserialize into a *bounded*
                     event queue (back-pressure to the TCP peer when full).
* sender thread   -- one per link in async mode (see ``outbox.py``).

Training semantics are unchanged from V1: all microbatches of a step use the
same parameters, gradients accumulate over the M microbatches (loss scaled
by 1/M) and every stage steps its optimizer exactly once per global step.
GPipe and 1F1B only reorder work inside that step.

If nothing happens for ``timeout_s`` the stage raises with a state dump
(current action, queues, buffered messages, saved contexts) instead of
hanging.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from meshtrain.models.base import ModelSpec
from meshtrain.networking.transport import Transport, TransportClosed, TransportTimeout
from meshtrain.runtime.distributed_autograd import BoundaryError, MicrobatchContext
from meshtrain.runtime.outbox import Outbox, OutboxError
from meshtrain.runtime.scheduler import ActionKind, Event, StageStateMachine
from meshtrain.runtime.serialization import bytes_to_tensor
from meshtrain.runtime.stage import Stage
from meshtrain.runtime.tensor_packet import MessageType, TensorPacket
from meshtrain.runtime.timeline import Timeline
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
    # V1.5
    schedule: str = "gpipe"                 # gpipe | 1f1b
    max_inflight_microbatches: int | None = None  # 1f1b: cap on saved microbatch graphs
    async_transport: bool = False           # False = V1 blocking sends on the compute thread
    max_outbound_queue: int | None = None   # default 2*M + 4 messages per link
    trace: bool = True                      # keep every timeline span (else only the current step)
    pinned_memory: bool = True              # CUDA: page-locked staging buffers
    buffer_pool: bool = True                # reuse staging / receive buffers
    correctness_probe_steps: int = 0        # record per-parameter grad norms (+ first-update deltas)


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
    timeline: list[dict] | None = None


class Inbox:
    """Reader thread per link -> one bounded event queue shared with the outbox."""

    def __init__(self, links: dict[str, Transport], timeline: Timeline, maxsize: int, poll_s: float = 0.2):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.timeline = timeline
        self.peak_depth = 0
        self._stop = threading.Event()
        for name, link in links.items():
            threading.Thread(target=self._reader, args=(name, link, poll_s), daemon=True,
                             name=f"recv-{name}").start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self.queue.put(item, timeout=0.2)
                self.peak_depth = max(self.peak_depth, self.queue.qsize())
                return True
            except queue.Full:
                continue  # back-pressure: stop reading from the socket until the executor drains
        return False

    def _reader(self, name: str, link: Transport, poll_s: float) -> None:
        tl = self.timeline
        while not self._stop.is_set():
            try:
                packet = link.recv_packet(timeout=poll_s)
            except TransportTimeout:
                continue
            except Exception as exc:  # closed peer, malformed frame, ...
                if not self._stop.is_set():
                    self._put(("error", name, exc))
                return
            end = tl.now()
            frame_s = getattr(link, "last_frame_s", 0.0)
            tl.add("NETWORK_RECV", end - frame_s, end, packet.step_id, packet.microbatch_id, link=name,
                   bytes=len(packet.payload))
            tensor = getattr(packet, "recv_buffer", None)  # received in place: nothing to decode
            if tensor is None and packet.has_tensor:
                with tl.span("DESERIALIZE", packet.step_id, packet.microbatch_id):
                    tensor = bytes_to_tensor(packet.payload, packet.dtype, packet.shape)
            if not self._put(("packet", name, packet, tensor)):
                return

    def stop(self) -> None:
        self._stop.set()


class _StepBuffers:
    def __init__(self):
        self.acts: dict[int, tuple] = {}          # mb -> (h2d handle, release token)
        self.targets: dict[int, torch.Tensor] = {}
        self.grads: dict[int, tuple] = {}         # mb -> (h2d handle, release token, meta)
        self.tokens: dict[int, list] = {}         # mb -> buffers to release after its backward


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
    timeline: Timeline | None = None,
) -> StageResult:
    logger = logger or EventLogger(worker, settings.job_id)
    if (upstream is None) != stage.is_first or (downstream is None) != stage.is_last:
        raise PipelineError("stage links do not match stage position")
    M = settings.num_microbatches
    if settings.batch_size % M:
        raise PipelineError("batch_size must be divisible by num_microbatches")
    S, sidx = stage.num_stages, stage.stage_index
    links = {k: v for k, v in ((UPSTREAM, upstream), (DOWNSTREAM, downstream)) if v is not None}
    tl = timeline or Timeline(worker, sidx)
    max_out = settings.max_outbound_queue or (2 * M + 4)
    inbox = Inbox(links, tl, maxsize=4 * M + 16) if links else None
    events = inbox.queue if inbox else queue.Queue()
    outbox = Outbox(links, tl, async_mode=settings.async_transport, max_queue=max_out,
                    timeout_s=settings.timeout_s, events=events)
    result = StageResult(sidx, worker)
    if capture_params:
        result.initial_params = stage.named_parameters_cpu()
    dev = stage.device
    dev.configure_transfers(pinned=settings.pinned_memory, pool=settings.buffer_pool)
    _recv_types = (int(MessageType.FORWARD_ACTIVATION), int(MessageType.BACKWARD_GRADIENT))
    for _link in links.values():  # receive activations/gradients straight into pooled buffers
        _link.payload_allocator = (lambda mtype, dtype, shape:
                                   dev.recv_buffer(shape, dtype) if mtype in _recv_types else None)
    future: dict[int, list] = {}          # buffered (link, packet, tensor) for later steps
    st: dict = {}                          # current step state: sm, bufs, step

    def fail_dump(reason: str) -> PipelineError:
        dump = {
            "reason": reason, "worker": worker, "stage": sidx,
            "state": st["sm"].snapshot() if st.get("sm") else None,
            "buffered": {k: sorted(getattr(st["bufs"], k)) for k in ("acts", "targets", "grads")} if st.get("bufs") else None,
            "future_steps": {s: len(v) for s, v in future.items()},
            "saved_contexts": [list(k) for k in stage.contexts.pending()],
            "pending_sends": outbox.pending(),
            "inbound_queue": events.qsize(),
        }
        logger.log("PIPELINE_STALLED", None, None, dump=json.dumps(dump))
        return PipelineError(f"{reason}; state: {json.dumps(dump)}")

    # -- inbound routing ------------------------------------------------------
    def apply(link: str, packet: TensorPacket, tensor) -> None:
        sm, bufs = st["sm"], st["bufs"]
        mb, mtype = packet.microbatch_id, packet.message_type
        if mtype == MessageType.TARGET and link == UPSTREAM:
            bufs.targets[mb] = tensor
            if mb in bufs.acts:
                sm.on(Event.ACTIVATION_RECEIVED, mb)
        elif mtype == MessageType.FORWARD_ACTIVATION and link == UPSTREAM:
            if mb in bufs.acts:
                raise BoundaryError(f"duplicate activation for step {packet.step_id} microbatch {mb}")
            bufs.acts[mb] = begin_h2d(tensor, packet.step_id, mb)  # prefetch to the device now
            if not stage.is_last or mb in bufs.targets:
                sm.on(Event.ACTIVATION_RECEIVED, mb)
        elif mtype == MessageType.BACKWARD_GRADIENT and link == DOWNSTREAM:
            bufs.grads[mb] = (*begin_h2d(tensor, packet.step_id, mb), dict(packet.meta))
            sm.on(Event.GRADIENT_RECEIVED, mb)
        else:
            raise BoundaryError(f"unexpected {mtype.name} from {link}")

    def pump(timeout: float) -> bool:
        """Handle one event; False if none arrived within ``timeout``."""
        if stop_event is not None and stop_event.is_set():
            raise PipelineError("job stopped")
        try:
            item = events.get(timeout=timeout)
        except queue.Empty:
            return False
        kind = item[0]
        if kind == "error":
            _, name, exc = item
            if isinstance(exc, TransportClosed):
                raise PipelineError(f"{name} peer disconnected: {exc}") from exc
            raise PipelineError(f"{name} link failed: {exc}") from exc
        if kind == "send_error":
            _, name, exc = item
            raise PipelineError(f"send to {name} peer failed: {type(exc).__name__}: {exc}") from exc
        if kind == "send_done":
            _, _name, step_id, mb = item
            if st.get("sm") is not None and step_id == st["step"]:
                st["sm"].on(Event.SEND_COMPLETED, mb)
            return True
        _, link, packet, tensor = item
        if packet.message_type == MessageType.ERROR:
            raise PipelineError(f"{link} peer reported error: {packet.meta.get('error')}")
        if packet.job_id != settings.job_id:
            raise BoundaryError(f"packet for job {packet.job_id!r} received in job {settings.job_id!r}")
        st["bytes_received"] += len(packet.payload)
        if packet.message_type == MessageType.TARGET and not stage.is_last:
            outbox.submit_packet(DOWNSTREAM, packet)  # relay unchanged
            return True
        step_id = packet.step_id
        if step_id == st["step"]:
            apply(link, packet, tensor)
        elif step_id == st["step"] + 1:
            future.setdefault(step_id, []).append((link, packet, tensor))
        else:
            raise BoundaryError(f"packet for step {step_id} while in step {st['step']}")
        return True

    def wait_until_ready(action, step: int) -> None:
        sm = st["sm"]
        if sm.ready(action):
            return
        cat = "WAIT_FORWARD" if action.kind == ActionKind.FORWARD else "WAIT_BACKWARD"
        t0 = tl.now()
        last_progress = time.monotonic()
        while not sm.ready(action):
            if pump(min(settings.timeout_s, 0.5)):
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > settings.timeout_s:
                raise fail_dump(f"pipeline stalled: no progress within network timeout {settings.timeout_s}s "
                                f"waiting for {action}")
        tl.add(cat, t0, tl.now(), step, action.microbatch)

    def begin_h2d(tensor: torch.Tensor, step: int, mb: int):
        if dev.backend == "cpu":
            return dev.begin_h2d(tensor)
        with tl.span("H2D_COPY", step, mb, phase="begin"):
            return dev.begin_h2d(tensor)

    def finish_h2d(entry, step: int, mb: int) -> torch.Tensor:
        handle, token = entry[0], entry[1]
        st["bufs"].tokens.setdefault(mb, []).append(token)
        if dev.backend == "cpu":
            return dev.finish_h2d(handle)
        with tl.span("H2D_COPY", step, mb, phase="finish"):
            return dev.finish_h2d(handle)

    def release_mb(mb: int) -> None:
        for token in st["bufs"].tokens.pop(mb, []):
            dev.after_use(token)

    try:
        for local_step in range(settings.steps):
            step = settings.step_offset + local_step
            t_step0 = tl.now()
            if stage.is_first:
                dev.reset_peak_memory()
            sm = StageStateMachine(sidx, S, M, step, settings.schedule, settings.max_inflight_microbatches)
            st.update(sm=sm, bufs=_StepBuffers(), step=step, bytes_received=st.get("bytes_received", 0))
            bytes_recv0, sent0 = st["bytes_received"], dict(outbox.bytes_sent)
            for link, packet, tensor in future.pop(step, []):
                apply(link, packet, tensor)
            if stage.is_first:
                x, y = spec.make_batch(step, settings.batch_size)
                xs, ys = x.chunk(M), y.chunk(M)
            loss_sum = 0.0
            peak_saved_bytes = peak_saved_mbs = 0
            bufs = st["bufs"]

            while not sm.done:
                action = sm.next_action()
                mb = action.microbatch
                wait_until_ready(action, step)
                if action.kind == ActionKind.FORWARD:
                    if stage.is_first:
                        inp = xs[mb]
                        if not stage.is_last:
                            outbox.submit_tensor(DOWNSTREAM, ys[mb], MessageType.TARGET, device=_HOST, step=step,
                                                 microbatch=mb, job_id=settings.job_id, source_worker=worker)
                    else:
                        inp = finish_h2d(bufs.acts.pop(mb), step, mb)
                    ctx = MicrobatchContext(step, mb)
                    sm.on(Event.FORWARD_STARTED, mb)
                    with tl.span("FORWARD_COMPUTE", step, mb):
                        out = stage.forward(inp, ctx)
                    sm.on(Event.FORWARD_FINISHED, mb)
                    peak_saved_bytes = max(peak_saved_bytes, stage.contexts.saved_bytes())
                    peak_saved_mbs = max(peak_saved_mbs, len(stage.contexts))
                    if not stage.is_last:
                        outbox.submit_tensor(DOWNSTREAM, out.detach(), MessageType.FORWARD_ACTIVATION, device=dev,
                                             step=step, microbatch=mb, notify=True, job_id=settings.job_id,
                                             source_worker=worker)
                    if settings.log_microbatch_events:
                        logger.log("FORWARD_COMPLETE", step, mb, output=out.numel() * out.element_size(),
                                   compute_s=ctx.timings["forward"])
                else:
                    sm.on(Event.BACKWARD_STARTED, mb)
                    if stage.is_last:
                        target = ys[mb] if stage.is_first else bufs.targets.pop(mb)
                        with tl.span("BACKWARD_COMPUTE", step, mb):
                            loss, grad_in, ctx = stage.loss_backward(target, (step, mb), loss_scale=1.0 / M)
                        meta = {"loss": float(loss)}
                    else:
                        entry = bufs.grads.pop(mb)
                        meta = entry[2]
                        grad = finish_h2d(entry, step, mb)
                        with tl.span("BACKWARD_COMPUTE", step, mb):
                            grad_in, ctx = stage.backward(grad, (step, mb))
                    sm.on(Event.BACKWARD_FINISHED, mb)
                    release_mb(mb)  # received activation/gradient buffers can be reused now
                    loss_sum += float(meta.get("loss", 0.0))
                    if not stage.is_first:
                        outbox.submit_tensor(UPSTREAM, grad_in, MessageType.BACKWARD_GRADIENT, device=dev,
                                             step=step, microbatch=mb, job_id=settings.job_id,
                                             source_worker=worker, meta=meta)
                    if settings.log_microbatch_events:
                        logger.log("BACKWARD_COMPLETE", step, mb, compute_s=ctx.timings["backward"])
                sm.advance()

            if not sm.all_complete():
                raise fail_dump("schedule finished with incomplete microbatches")
            if settings.capture_gradients_at_step == step:
                result.gradients = stage.named_gradients()
            probe = local_step < settings.correctness_probe_steps
            if probe:
                grad_norms = {n: float(p.grad.detach().float().norm()) for n, p in stage.module.named_parameters()
                              if p.grad is not None}
                before = stage.named_parameters_cpu() if local_step == 0 else None
            stage.refresh_tensor_store()
            mem = stage.memory_report()  # gradients present, before zero_grad
            mem["tensor_store"] = stage.tensor_store.bytes_by_role()
            mem["saved_activations_peak"] = peak_saved_bytes
            mem["saved_microbatches_peak"] = peak_saved_mbs
            with tl.span("OPTIMIZER_STEP", step):
                opt_s = stage.optimizer_step()
            stage.zero_grad()
            if probe:
                correctness = {"grad_norms": grad_norms}
                if before is not None:
                    after = stage.named_parameters_cpu()
                    correctness["param_delta_norms"] = {n: float((after[n] - before[n]).float().norm())
                                                        for n in before}
            t_step1 = tl.now()
            step_s = t_step1 - t_step0
            m = tl.step_metrics(step, (t_step0, t_step1))
            phase = m["phase_s"]
            mean_loss = loss_sum / M
            dev_mem = dev.memory_stats()
            bytes_sent = sum(outbox.bytes_sent.values()) - sum(sent0.values())
            record = {
                "event": "STEP_COMPLETE", "job": settings.job_id, "worker": worker,
                "stage": sidx, "backend": dev.backend, "step": step, "schedule": settings.schedule,
                "async_transport": settings.async_transport,
                "step_s": step_s, "forward_s": phase.get("FORWARD_COMPUTE", 0.0),
                "backward_s": phase.get("BACKWARD_COMPUTE", 0.0), "optimizer_s": opt_s,
                "compute_s": m["compute_s"], "comm_s": m["communication_s"],
                "communication_s": m["communication_s"], "overlapped_s": m["overlapped_s"],
                "exposed_communication_s": m["exposed_communication_s"], "overlap_ratio": m["overlap_ratio"],
                "idle_s": m["idle_s"],
                "wait_forward_s": phase.get("WAIT_FORWARD", 0.0), "wait_backward_s": phase.get("WAIT_BACKWARD", 0.0),
                "bytes_sent": bytes_sent, "bytes_received": st["bytes_received"] - bytes_recv0,
                "lifecycle": {k: v for k, v in phase.items() if not k.startswith("WAIT")},
                "outbound_queue_peak": max(outbox.peak_depth.values(), default=0),
                "inbound_queue_peak": inbox.peak_depth if inbox else 0,
                "peak_inflight_microbatches": sm.peak_inflight,
                "buffer_pool": dev.host_pool.stats(),
                "utilization": m["compute_s"] / step_s if step_s else 0.0,
                "memory": {**mem, "device_allocated": dev_mem.get("allocated", 0),
                           "device_peak": dev_mem.get("peak_allocated", 0), "device_total": dev_mem.get("total", 0)},
            }
            if probe:
                record["correctness"] = correctness  # local parameter names (layer index within the stage)
            if stage.is_first:
                record["loss"] = mean_loss
                record["samples_per_s"] = settings.batch_size / step_s if step_s else 0.0
                result.losses.append(mean_loss)
            result.step_metrics.append(record)
            if metrics_callback is not None:
                metrics_callback(record)
            if not settings.trace:
                tl.drop_before(t_step1)
            if local_step % settings.log_every == 0 or local_step == settings.steps - 1:
                fields = {"step_s": step_s, "compute_s": m["compute_s"], "comm_s": m["communication_s"],
                          "exposed_s": m["exposed_communication_s"], "idle_s": m["idle_s"], "sent_bytes": bytes_sent}
                if stage.is_first:
                    fields = {"loss": mean_loss, **fields}
                logger.log("STEP_COMPLETE", step, None, **fields)
        outbox.flush()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.log("ERROR", None, None, error=result.error[:500])
        # Best effort: tell neighbours so they fail fast instead of timing out.
        for link in links.values():
            try:
                link.send_packet(TensorPacket(MessageType.ERROR, job_id=settings.job_id, source_worker=worker,
                                              meta={"error": f"{worker}: {result.error}"[:200]}))
            except Exception:
                pass
        if isinstance(exc, OutboxError):
            raise PipelineError(str(exc)) from exc
        raise
    finally:
        if inbox is not None:
            inbox.stop()
        outbox.close()
    if capture_params:
        result.final_params = stage.named_parameters_cpu()
    if settings.trace:
        result.timeline = tl.export()
    return result


class _HostAdapter:
    """Staging for tensors that already live in host memory (targets)."""

    @staticmethod
    def begin_d2h(tensor: torch.Tensor):
        t = tensor.detach()
        return lambda: (t, None)


_HOST = _HostAdapter()

"""Worker-side execution: the data-plane listener and the stage executor.

``DataPlaneServer`` owns the worker's single TCP listening port. Incoming
connections start with a HELLO packet and are routed:

* ``command_kind == "PROBE"``  -> answered by the network profiler
* ``job_id`` + ``stage``       -> parked until the stage executor for that job
                                  claims its upstream link

``run_assignment`` builds the stage (only its own layers), wires up links
and runs the pipeline loop; it is shared by the worker agent and the
coordinator-free ``meshtrain stage`` command.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from meshtrain.config import parse_config
from meshtrain.networking.tcp import TCPListener, TCPTransport, connect
from meshtrain.networking.transport import TransportTimeout
from meshtrain.profiler.network import serve_probe
from meshtrain.runtime.memory_check import (
    StageMemoryError,
    format_validation,
    is_oom,
    release_device_memory,
    validate_stage,
)
from meshtrain.runtime.pipeline import StageResult, run_stage
from meshtrain.runtime.stage import Stage
from meshtrain.telemetry import EventLogger
from meshtrain.worker.device import DeviceAdapter


class DataPlaneServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 29500, *, token: str | None = None,
                 max_payload_bytes: int = 2 * 1024**3, logger: EventLogger | None = None, tensor_server=None):
        self.listener = TCPListener(host, port, token=token, max_payload_bytes=max_payload_bytes)
        self.tensor_server = tensor_server   # V2.5: this worker lends RAM (networking/tensor_server.py)
        self.port = self.listener.port
        self.logger = logger
        self._pending: dict[tuple[str, int], TCPTransport] = {}
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="dataplane-accept")

    def start(self) -> "DataPlaneServer":
        self._thread.start()
        return self

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                link, hello = self.listener.accept(timeout=0.5)
            except TransportTimeout:
                continue
            except Exception as exc:  # bad hello / token: drop the connection, keep serving
                if self._stop.is_set():
                    return
                if self.logger:
                    self.logger.log("DATAPLANE_REJECTED", error=str(exc))
                continue
            if hello.get("command_kind") == "PROBE":
                threading.Thread(target=self._probe, args=(link,), daemon=True).start()
            elif hello.get("command_kind") == "TENSOR_STORE":
                if self.tensor_server is None:
                    link.close()   # this worker does not lend RAM
                else:
                    threading.Thread(target=self.tensor_server.handle, args=(link, self._stop), daemon=True,
                                     name="tensor-store").start()
            elif "job_id" in hello and "stage" in hello:
                with self._cond:
                    key = (str(hello["job_id"]), int(hello["stage"]))
                    self._pending[key] = link
                    self._cond.notify_all()
            else:
                link.close()

    def _probe(self, link: TCPTransport) -> None:
        try:
            serve_probe(link)
        except Exception:
            pass
        finally:
            link.close()

    def claim(self, job_id: str, from_stage: int, timeout: float, stop_event=None) -> TCPTransport:
        deadline = time.monotonic() + timeout
        with self._cond:
            while (job_id, from_stage) not in self._pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TransportTimeout(f"stage {from_stage} of job {job_id} never connected "
                                           f"(waited {timeout:.0f}s)")
                if stop_event is not None and stop_event.is_set():
                    raise TransportTimeout("claim cancelled: job stopped")
                self._cond.wait(min(left, 0.5))
            return self._pending.pop((job_id, from_stage))

    def close(self) -> None:
        self._stop.set()
        self.listener.close()


def run_assignment(
    assignment: dict,
    *,
    device: DeviceAdapter,
    dataplane: DataPlaneServer,
    worker_name: str,
    token: str | None = None,
    metrics_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    runs_dir: str = "runs",
    verbose: bool = True,
) -> StageResult:
    """Execute one START_STAGE assignment (see coordinator.server.start_job).

    Startup: materialise the stage, validate its memory against the planner's
    estimate (``memory_check``), then connect links and train. Any memory
    failure before training is raised as ``StageMemoryError`` with the
    measurements, after releasing the device memory, so the coordinator can
    replan.
    """
    cfg = parse_config(assignment["config"])
    job_id = assignment["job_id"]
    data_job = assignment.get("data_job_id", job_id)  # unique per placement attempt
    idx, n = int(assignment["stage_index"]), int(assignment["num_stages"])
    start, end = assignment["layers"]
    run_dir = Path(runs_dir) / job_id
    logger = EventLogger(worker_name, job_id, jsonl_path=run_dir / f"events-{worker_name}.jsonl", verbose=verbose)
    spec = cfg.build_model_spec()
    up = down = None
    stage = None
    estimate = assignment.get("memory_estimate")
    emit = metrics_callback or (lambda rec: None)
    try:
        logger.log("STAGE_ASSIGNED", stage=idx, layers=f"{start}-{end - 1}", device=str(device.device),
                   attempt=assignment.get("attempt", 0))
        try:
            policy = cfg.residency_policy()
            budget = cfg.memory.budget_bytes(device.memory_total(), device.backend)
            if policy.active:
                from meshtrain.planner.residency import resolve_stage_policy

                M = cfg.training.num_microbatches
                policy, rplan = resolve_stage_policy(policy, spec, start, end,
                                                     microbatch_size=cfg.training.batch_size // M, budget=budget,
                                                     optimizer=cfg.training.optimizer, backend=device.backend,
                                                     num_microbatches=M, schedule=cfg.pipeline.schedule,
                                                     stage_index=idx, num_stages=n)
                if rplan is not None:
                    logger.log("RESIDENCY_PLAN", stage=idx, hot=len(rplan.hot_groups),
                               cold=len(rplan.groups) - len(rplan.hot_groups),
                               device_gb=round(rplan.device_total / 1024**3, 3))
            stage = Stage(spec.build_stage(start, end), stage_index=idx, num_stages=n, device=device,
                          optimizer=cfg.training.optimizer, lr=cfg.training.learning_rate, loss_fn=spec.loss_fn,
                          name=worker_name, layer_offset=start, residency=policy, accelerator_budget=budget)
            report = None
            if cfg.memory.validate_runtime_usage:
                sf = cfg.memory.safety_factors().get(device.backend, cfg.memory.safety_factor)
                report = validate_stage(stage, device, estimate, probe=cfg.memory.probe_allocation, safety_factor=sf)
                logger.log("MEMORY_VALIDATED", stage=idx, summary=format_validation(report))
                emit({"event": "MEMORY_VALIDATION", "job": job_id, "worker": worker_name, "stage": idx,
                      "phase": "startup", **report})
        except Exception as exc:
            if not is_oom(exc):
                raise
            rep = getattr(exc, "report", {"available": device.available_memory()})
            stage = None
            release_device_memory(device)
            logger.log("STAGE_MEMORY_FAILED", stage=idx, error=str(exc)[:300])
            raise StageMemoryError(f"startup: {exc}", rep) from exc

        timeout = cfg.network.timeout_s
        max_bytes = int(cfg.network.max_tensor_mb * 1024**2)
        if assignment.get("downstream"):
            d = assignment["downstream"]
            hello = {"job_id": data_job, "stage": idx}
            if token is not None:
                hello["token"] = token
            down = connect(d["host"], int(d["port"]), timeout=cfg.network.connect_timeout_s, hello=hello,
                           frame_timeout_s=timeout, max_payload_bytes=max_bytes, stop_event=stop_event)
            logger.log("LINK_CONNECTED", direction="downstream", peer=f"{d['host']}:{d['port']}")
        if idx > 0:
            up = dataplane.claim(data_job, idx - 1, timeout=cfg.network.connect_timeout_s, stop_event=stop_event)
            up.frame_timeout_s = timeout
            up.max_payload_bytes = max_bytes
            logger.log("LINK_CONNECTED", direction="upstream", peer=up.peer)
        settings = cfg.pipeline_settings(data_job, log_microbatch_events=bool(assignment.get("trace", False)),
                                         trace=bool(assignment.get("timeline", True)))

        first = {"done": False}

        def on_step(rec: dict) -> None:
            if not first["done"] and estimate:
                first["done"] = True
                peak = int(rec["memory"].get("device_peak", 0))
                rec["memory_validation"] = {
                    "estimated_required": int(estimate["required"]), "actual_peak_step0": peak,
                    "peak_estimate_error": (peak - estimate["required"]) / estimate["required"]
                    if estimate["required"] else 0.0,
                    "note": "CPU peak is process RSS" if device.backend == "cpu" else "torch allocator peak"}
            rec["job"] = job_id
            emit(rec)

        result = run_stage(stage, spec, settings, upstream=up, downstream=down, worker=worker_name, logger=logger,
                           metrics_callback=on_step, stop_event=stop_event)
        if result.timeline is not None:
            import json as _json

            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / f"timeline-stage{idx}-{worker_name}.json").write_text(_json.dumps(result.timeline))
        return result
    finally:
        for link in (up, down):
            if link is not None:
                link.close()
        if stage is not None:
            stage.close()   # hooks and the allocator cap (this worker runs later jobs too)
        stage = None
        release_device_memory(device)
        logger.close()

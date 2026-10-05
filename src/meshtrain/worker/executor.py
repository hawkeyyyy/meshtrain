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
from meshtrain.runtime.pipeline import PipelineSettings, StageResult, run_stage
from meshtrain.runtime.stage import Stage
from meshtrain.telemetry import EventLogger
from meshtrain.worker.device import DeviceAdapter


class DataPlaneServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 29500, *, token: str | None = None,
                 max_payload_bytes: int = 2 * 1024**3, logger: EventLogger | None = None):
        self.listener = TCPListener(host, port, token=token, max_payload_bytes=max_payload_bytes)
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

    def claim(self, job_id: str, from_stage: int, timeout: float) -> TCPTransport:
        deadline = time.monotonic() + timeout
        with self._cond:
            while (job_id, from_stage) not in self._pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TransportTimeout(f"stage {from_stage} of job {job_id} never connected "
                                           f"(waited {timeout:.0f}s)")
                self._cond.wait(left)
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
    """Execute one START_STAGE assignment (see coordinator.server.start_job)."""
    cfg = parse_config(assignment["config"])
    job_id = assignment["job_id"]
    idx, n = int(assignment["stage_index"]), int(assignment["num_stages"])
    start, end = assignment["layers"]
    run_dir = Path(runs_dir) / job_id
    logger = EventLogger(worker_name, job_id, jsonl_path=run_dir / f"events-{worker_name}.jsonl", verbose=verbose)
    spec = cfg.build_model_spec()
    up = down = None
    try:
        logger.log("STAGE_ASSIGNED", stage=idx, layers=f"{start}-{end - 1}", device=str(device.device))
        stage = Stage(spec.build_stage(start, end), stage_index=idx, num_stages=n, device=device,
                      optimizer=cfg.training.optimizer, lr=cfg.training.learning_rate, loss_fn=spec.loss_fn,
                      name=worker_name)
        timeout = cfg.network.timeout_s
        max_bytes = int(cfg.network.max_tensor_mb * 1024**2)
        if assignment.get("downstream"):
            d = assignment["downstream"]
            hello = {"job_id": job_id, "stage": idx}
            if token is not None:
                hello["token"] = token
            down = connect(d["host"], int(d["port"]), timeout=cfg.network.connect_timeout_s, hello=hello,
                           frame_timeout_s=timeout, max_payload_bytes=max_bytes)
            logger.log("LINK_CONNECTED", direction="downstream", peer=f"{d['host']}:{d['port']}")
        if idx > 0:
            up = dataplane.claim(job_id, idx - 1, timeout=cfg.network.connect_timeout_s)
            up.frame_timeout_s = timeout
            up.max_payload_bytes = max_bytes
            logger.log("LINK_CONNECTED", direction="upstream", peer=up.peer)
        settings = PipelineSettings(
            job_id=job_id, steps=cfg.training.steps, batch_size=cfg.training.batch_size,
            num_microbatches=cfg.training.num_microbatches, timeout_s=timeout,
            log_every=cfg.training.log_every, log_microbatch_events=bool(assignment.get("trace", False)),
        )
        return run_stage(stage, spec, settings, upstream=up, downstream=down, worker=worker_name, logger=logger,
                         metrics_callback=metrics_callback, stop_event=stop_event)
    finally:
        for link in (up, down):
            if link is not None:
                link.close()
        logger.close()

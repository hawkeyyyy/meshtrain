"""Worker agent: ``meshtrain worker join <coordinator>``.

1. detect hardware, pick a device adapter (cuda -> mps -> cpu, or forced)
2. start the data-plane listener
3. register with the coordinator (cluster token), start heartbeats
4. long-poll commands: RUN_BENCHMARK, PROBE_NETWORK, START_STAGE, STOP_JOB
"""

from __future__ import annotations

import socket
import threading
import time
import traceback

from meshtrain.networking.control import ControlClient, ControlError
from meshtrain.profiler.benchmark import benchmark_device
from meshtrain.profiler.hardware import detect_hardware
from meshtrain.profiler.network import measure_link
from meshtrain.telemetry import EventLogger
from meshtrain.worker.device import DeviceAdapter, select_device
from meshtrain.worker.executor import DataPlaneServer, run_assignment
from meshtrain.worker.heartbeat import Heartbeat


def guess_advertise_host(coordinator_base: str) -> str:
    """The local IP used to reach the coordinator (what peers should dial)."""
    host = coordinator_base.split("://", 1)[-1].rsplit(":", 1)[0]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 9))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class WorkerAgent:
    def __init__(self, coordinator: str, token: str, *, device: str | None = None, name: str | None = None,
                 data_port: int = 29500, bind_host: str = "0.0.0.0", advertise_host: str | None = None,
                 runs_dir: str = "runs", verbose: bool = True, quick_benchmark: bool = False):
        self.client = ControlClient(coordinator, token)
        self.token = token
        self.device: DeviceAdapter = select_device(device)
        self.hardware = detect_hardware()
        self.name = name or self.hardware["hostname"]
        self.log = EventLogger(self.name, "-", verbose=verbose)
        self.verbose = verbose
        self.runs_dir = runs_dir
        self.quick_benchmark = quick_benchmark
        self.dataplane = DataPlaneServer(bind_host, data_port, token=token, logger=self.log).start()
        self.advertise_host = advertise_host or guess_advertise_host(self.client.base)
        self.worker_id: str | None = None
        self._stop = threading.Event()
        self._jobs: dict[str, threading.Event] = {}
        self.heartbeat: Heartbeat | None = None

    # -- registration -----------------------------------------------------
    def register(self) -> str:
        resp = self.client.register({
            "name": self.name, "hardware": self.hardware, "backend": self.device.backend,
            "device": self.device.describe(), "data_host": self.advertise_host, "data_port": self.dataplane.port,
        })
        self.worker_id = resp["worker_id"]
        self.log.worker = self.worker_id
        self.log.log("REGISTERED", coordinator=self.client.base, backend=self.device.backend,
                     device=self.device.name(), data=f"{self.advertise_host}:{self.dataplane.port}")
        if self.heartbeat is None:
            self.heartbeat = Heartbeat(self.client, lambda: self.worker_id, self._memory,
                                       interval_s=resp.get("heartbeat_interval_s", 3.0),
                                       on_unknown=self._reregister,
                                       on_error=lambda e: self.log.log("HEARTBEAT_FAILED", error=str(e))).start()
        return self.worker_id

    def _reregister(self) -> None:
        self.log.log("REREGISTER", reason="coordinator does not know this worker")
        try:
            self.register()
        except ControlError as exc:
            self.log.log("REGISTER_FAILED", error=str(exc))

    def _memory(self) -> dict:
        try:
            return {k: v for k, v in self.device.memory_stats().items() if isinstance(v, (int, bool))}
        except Exception:
            return {}

    # -- command loop -----------------------------------------------------
    def run_forever(self) -> None:
        while self.worker_id is None and not self._stop.is_set():
            try:
                self.register()
            except ControlError as exc:
                self.log.log("REGISTER_FAILED", error=str(exc), retry_in_s=3)
                self._stop.wait(3)
        while not self._stop.is_set():
            try:
                commands = self.client.poll_commands(self.worker_id, wait_s=10.0)
            except ControlError as exc:
                if "404" in str(exc):
                    self._reregister()
                else:
                    self.log.log("COORDINATOR_UNREACHABLE", error=str(exc)[:200])
                    self._stop.wait(2)
                continue
            for cmd in commands:
                threading.Thread(target=self._handle, args=(cmd,), daemon=True,
                                 name=f"cmd-{cmd.get('type')}").start()

    def stop(self) -> None:
        self._stop.set()
        for ev in self._jobs.values():
            ev.set()
        if self.heartbeat:
            self.heartbeat.stop()
        self.dataplane.close()

    def _event(self, type_: str, data: dict) -> None:
        try:
            self.client.event(self.worker_id, type_, data)
        except ControlError as exc:
            self.log.log("EVENT_POST_FAILED", type=type_, error=str(exc)[:200])

    def _handle(self, cmd: dict) -> None:
        kind = cmd.get("type")
        rid = cmd.get("request_id")
        try:
            if kind == "RUN_BENCHMARK":
                self.log.log("BENCHMARK_START")
                result = benchmark_device(self.device, quick=self.quick_benchmark)
                self.log.log("BENCHMARK_DONE", gflops=round(result["matmul_gflops"], 1))
                self._event("benchmark_result", {"request_id": rid, "result": result})
            elif kind == "PROBE_NETWORK":
                result = measure_link(cmd["host"], int(cmd["port"]), token=self.token, pings=int(cmd.get("pings", 10)),
                                      payload_mb=float(cmd.get("payload_mb", 16)))
                self.log.log("PROBE_DONE", target=cmd.get("target"), latency_s=result["latency_s"],
                             mbps=round(result["bandwidth_Mbps"], 1))
                self._event("probe_result", {"request_id": rid, "result": result})
            elif kind == "START_STAGE":
                self._run_stage(cmd)
            elif kind == "STOP_JOB":
                ev = self._jobs.get(cmd["job_id"])
                if ev:
                    self.log.log("STOP_REQUESTED", job_id=cmd["job_id"], reason=cmd.get("reason"))
                    ev.set()
            else:
                raise ValueError(f"unknown command {kind!r}")
        except Exception as exc:
            self.log.log("COMMAND_FAILED", type=kind, error=f"{type(exc).__name__}: {exc}")
            if rid:
                self._event("command_error", {"request_id": rid, "error": f"{type(exc).__name__}: {exc}"})

    def _run_stage(self, cmd: dict) -> None:
        job_id, idx = cmd["job_id"], cmd["stage_index"]
        stop = self._jobs.setdefault(job_id, threading.Event())
        t0 = time.time()
        try:
            result = run_assignment(
                cmd, device=self.device, dataplane=self.dataplane, worker_name=self.worker_id, token=self.token,
                metrics_callback=lambda rec: self._event("stage_metrics", {"job_id": job_id, "record": rec}),
                stop_event=stop, runs_dir=self.runs_dir, verbose=self.verbose)
            self._event("stage_done", {"job_id": job_id, "stage": idx,
                                       "summary": {"steps": len(result.step_metrics), "wall_s": time.time() - t0,
                                                   "final_loss": result.losses[-1] if result.losses else None}})
        except Exception as exc:
            if stop.is_set():
                self.log.log("STAGE_STOPPED", job_id=job_id)
                return
            self.log.log("STAGE_FAILED", job_id=job_id, error=f"{type(exc).__name__}: {exc}")
            if self.verbose:
                traceback.print_exc()
            self._event("stage_error", {"job_id": job_id, "stage": idx, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._jobs.pop(job_id, None)

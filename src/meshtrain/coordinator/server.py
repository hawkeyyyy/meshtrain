"""FastAPI coordinator (control plane only -- no tensors, no training).

Endpoints (all require header ``X-MeshTrain-Token`` except /health):

    POST /workers/register              worker hardware + data-plane address -> worker_id
    POST /workers/{id}/heartbeat        liveness + memory snapshot
    GET  /workers/{id}/commands         long poll for commands
    POST /workers/{id}/events           benchmark/probe results, stage metrics/done/error
    GET  /cluster/status                workers
    POST /cluster/benchmark             start compute + network benchmark
    GET  /cluster/benchmark             benchmark progress/results
    POST /plan                          dry-run placement for a config
    POST /jobs                          plan + start a training job
    GET  /jobs, GET /jobs/{id}          job status (losses, metrics)
    POST /jobs/{id}/stop                stop a job
"""

from __future__ import annotations

import hmac
import itertools
import json
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from meshtrain.config import ConfigError, parse_config
from meshtrain.coordinator.registry import WorkerRegistry
from meshtrain.coordinator.scheduler import plan_job
from meshtrain.coordinator.state import JobRecord, JobStatus, WorkerRecord, WorkerStatus
from meshtrain.planner.cost import NetworkModel
from meshtrain.profiler.benchmark import normalise_scores
from meshtrain.summary import format_summary, summarize
from meshtrain.telemetry import EventLogger, MetricsWriter, read_jsonl

MAX_REQUEST_BYTES = 8 * 1024 * 1024


class RegisterRequest(BaseModel):
    name: str
    hardware: dict
    backend: str
    device: dict
    data_host: str
    data_port: int


class HeartbeatRequest(BaseModel):
    memory: dict = {}
    status: str | None = None


class EventRequest(BaseModel):
    type: str
    data: dict = {}


class JobRequest(BaseModel):
    config: dict


class BenchmarkRequest(BaseModel):
    pings: int = 10
    payload_mb: float = 16.0
    network: bool = True


class Coordinator:
    def __init__(self, token: str, runs_dir: str = "runs", heartbeat_timeout_s: float = 15.0,
                 command_timeout_s: float = 120.0, verbose: bool = True):
        self.token = token
        self.runs_dir = Path(runs_dir)
        self.registry = WorkerRegistry(heartbeat_timeout_s)
        self.registry.on_offline.append(self._worker_offline)
        self.jobs: dict[str, JobRecord] = {}
        self.command_timeout_s = command_timeout_s
        self.log = EventLogger("coordinator", "-", verbose=verbose)
        self._lock = threading.RLock()
        self._waiters: dict[str, tuple[threading.Event, dict]] = {}
        self.benchmark: dict = {"status": "NEVER_RUN", "workers": {}, "links": {}}
        self._writers: dict[str, MetricsWriter] = {}
        self._stop = threading.Event()
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True, name="liveness-monitor")
        self._monitor.start()

    # -- liveness -------------------------------------------------------
    def _monitor_loop(self) -> None:
        while not self._stop.wait(0.5):
            self.registry.check_liveness()

    def _worker_offline(self, w: WorkerRecord) -> None:
        self.log.log("WORKER_OFFLINE", worker_id=w.worker_id,
                     silent_s=round(time.time() - w.last_heartbeat, 1))
        with self._lock:
            for job in self.jobs.values():
                if job.status == JobStatus.RUNNING and w.worker_id in job.stage_workers:
                    self._fail_job(job, f"worker {w.worker_id} went offline "
                                        f"(no heartbeat for {self.registry.heartbeat_timeout_s:.0f}s)")

    def shutdown(self) -> None:
        self._stop.set()

    # -- request/response waits -------------------------------------------
    def _request(self, worker_id: str, command: dict, timeout: float) -> dict | None:
        rid = uuid.uuid4().hex[:12]
        ev, box = threading.Event(), {}
        self._waiters[rid] = (ev, box)
        self.registry.send(worker_id, {**command, "request_id": rid})
        ok = ev.wait(timeout)
        self._waiters.pop(rid, None)
        return box if ok else None

    def _resolve(self, rid: str, data: dict) -> None:
        waiter = self._waiters.get(rid)
        if waiter:
            waiter[1].update(data)
            waiter[0].set()

    # -- benchmark ------------------------------------------------------
    def start_benchmark(self, req: BenchmarkRequest) -> dict:
        with self._lock:
            if self.benchmark.get("status") == "RUNNING":
                return self.benchmark
            workers = [w for w in self.registry.online() if w.status == WorkerStatus.ONLINE]
            self.benchmark = {"status": "RUNNING", "started_at": time.time(), "workers": {}, "links": {},
                              "errors": []}
        threading.Thread(target=self._run_benchmark, args=(workers, req), daemon=True).start()
        return self.benchmark

    def _run_benchmark(self, workers: list[WorkerRecord], req: BenchmarkRequest) -> None:
        bench = self.benchmark
        results: dict[str, dict] = {}

        def one(w):
            r = self._request(w.worker_id, {"type": "RUN_BENCHMARK"}, self.command_timeout_s)
            if r is None or "error" in r:
                bench["errors"].append(f"{w.worker_id}: {r.get('error') if r else 'benchmark timed out'}")
            else:
                results[w.worker_id] = r["result"]

        threads = [threading.Thread(target=one, args=(w,)) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        scores = normalise_scores(results)
        for wid, r in results.items():
            r["compute_score"] = scores[wid]
            rec = self.registry.get(wid)
            if rec:
                rec.benchmark = r
            bench["workers"][wid] = r
        if req.network:
            # Directional probes, one at a time so links do not compete.
            for src, dst in itertools.permutations(workers, 2):
                r = self._request(src.worker_id, {"type": "PROBE_NETWORK", "target": dst.worker_id,
                                                  "host": dst.data_host, "port": dst.data_port,
                                                  "pings": req.pings, "payload_mb": req.payload_mb},
                                  self.command_timeout_s)
                key = f"{src.worker_id}->{dst.worker_id}"
                if r is None or "error" in r:
                    bench["errors"].append(f"{key}: {r.get('error') if r else 'probe timed out'}")
                else:
                    bench["links"][key] = r["result"]
        bench["status"] = "DONE"
        bench["finished_at"] = time.time()
        self.log.log("BENCHMARK_DONE", workers=len(results), links=len(bench["links"]))

    def network_model(self) -> NetworkModel:
        return NetworkModel.from_measurements(self.benchmark.get("links", {}))

    # -- jobs -------------------------------------------------------------
    def plan(self, config: dict):
        cfg = parse_config(config)
        workers = [w for w in self.registry.online() if w.status == WorkerStatus.ONLINE]
        if not workers:
            raise ValueError("no idle online workers")
        return cfg, plan_job(cfg, workers, self.network_model())

    def start_job(self, config: dict) -> JobRecord:
        cfg, plan = self.plan(config)
        if not plan.stages or (not plan.feasible and cfg.placement.enforce_memory_check):
            raise ValueError("placement infeasible:\n" + plan.format())
        job_id = f"{cfg.job.name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        run_dir = self.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        layer_names = [cfg.build_model_spec().layer_name(i) for i in range(cfg.model_num_layers())]
        (run_dir / "plan.txt").write_text(plan.format(layer_names) + "\n")
        (run_dir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
        job = JobRecord(job_id, cfg.model_dump(mode="json"), plan.to_dict(), run_dir=str(run_dir),
                        stage_workers=[s.worker_id for s in plan.stages])
        self._writers[job_id] = MetricsWriter(run_dir / "metrics.jsonl")
        with self._lock:
            self.jobs[job_id] = job
            for i, s in enumerate(plan.stages):
                w = self.registry.get(s.worker_id)
                w.status, w.current_job = WorkerStatus.BUSY, job_id
            job.status = JobStatus.RUNNING
        n = len(plan.stages)
        for i, s in enumerate(plan.stages):
            nxt = self.registry.get(plan.stages[i + 1].worker_id) if i + 1 < n else None
            self.registry.send(s.worker_id, {
                "type": "START_STAGE", "job_id": job_id, "config": job.config, "stage_index": i,
                "num_stages": n, "layers": [s.start, s.end],
                "downstream": {"host": nxt.data_host, "port": nxt.data_port, "worker_id": nxt.worker_id} if nxt else None,
                "upstream_worker": plan.stages[i - 1].worker_id if i > 0 else None,
            })
        self.log.log("JOB_STARTED", None, None, job_id=job_id,
                     placement=" | ".join(f"{s.worker_name}:{s.start}-{s.end - 1}" for s in plan.stages))
        return job

    def _release(self, job: JobRecord) -> None:
        for wid in job.stage_workers:
            w = self.registry.get(wid)
            if w and w.current_job == job.job_id:
                w.current_job = None
                if w.status == WorkerStatus.BUSY:
                    w.status = WorkerStatus.ONLINE

    def _finish(self, job: JobRecord, status: JobStatus, error: str | None = None) -> None:
        job.status, job.error, job.finished_at = status, error, time.time()
        self._release(job)
        path = Path(job.run_dir) / "metrics.jsonl"
        if path.exists():
            s = summarize(read_jsonl(path))
            s["status"], s["error"] = status.value, error
            (Path(job.run_dir) / "summary.json").write_text(json.dumps(s, indent=2, default=str))
            (Path(job.run_dir) / "summary.txt").write_text(format_summary(s) + "\n")
            job.summary = s

    def _fail_job(self, job: JobRecord, error: str) -> None:
        if job.status != JobStatus.RUNNING:
            return
        self.log.log("JOB_FAILED", job_id=job.job_id, error=error)
        for wid in job.stage_workers:
            w = self.registry.get(wid)
            if w and w.status != WorkerStatus.OFFLINE:
                self.registry.send(wid, {"type": "STOP_JOB", "job_id": job.job_id, "reason": error})
        self._finish(job, JobStatus.FAILED, error)

    def stop_job(self, job_id: str) -> JobRecord:
        job = self.jobs[job_id]
        with self._lock:
            if job.status == JobStatus.RUNNING:
                for wid in job.stage_workers:
                    self.registry.send(wid, {"type": "STOP_JOB", "job_id": job_id, "reason": "stopped by user"})
                self._finish(job, JobStatus.STOPPED, "stopped by user")
        return job

    # -- worker events ----------------------------------------------------
    def handle_event(self, worker_id: str, ev: EventRequest) -> None:
        d = ev.data
        if ev.type in ("benchmark_result", "probe_result", "command_error"):
            self._resolve(d.get("request_id", ""), d)
            return
        job = self.jobs.get(d.get("job_id", ""))
        if job is None:
            return
        with self._lock:
            if ev.type == "stage_metrics":
                rec = d["record"]
                self._writers[job.job_id].write(rec)
                job.last_metrics[rec["stage"]] = rec
                if "loss" in rec:
                    job.losses.append((rec["step"], rec["loss"]))
            elif ev.type == "stage_done":
                job.stages_done[d["stage"]] = d.get("summary", {})
                if job.status == JobStatus.RUNNING and all(i in job.stages_done for i in range(len(job.stage_workers))):
                    self.log.log("JOB_COMPLETED", job_id=job.job_id)
                    self._finish(job, JobStatus.COMPLETED)
            elif ev.type == "stage_error":
                self._fail_job(job, f"stage {d.get('stage')} on {worker_id}: {d.get('error')}")


def create_app(coordinator: Coordinator) -> FastAPI:
    app = FastAPI(title="MeshTrain coordinator", version="1")
    app.state.coordinator = coordinator
    c = coordinator

    @app.middleware("http")
    async def limit_size(request: Request, call_next):
        size = request.headers.get("content-length")
        if size is not None and int(size) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    def auth(x_meshtrain_token: str | None = Header(default=None)):
        if x_meshtrain_token is None or not hmac.compare_digest(x_meshtrain_token, c.token):
            raise HTTPException(401, "invalid or missing cluster token")

    def worker_or_404(worker_id: str) -> WorkerRecord:
        w = c.registry.get(worker_id)
        if w is None:
            raise HTTPException(404, f"unknown worker {worker_id}")
        return w

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/workers/register", dependencies=[Depends(auth)])
    def register(req: RegisterRequest):
        w = c.registry.register(name=req.name, hardware=req.hardware, backend=req.backend,
                                device_info=req.device, data_host=req.data_host, data_port=req.data_port)
        c.log.log("WORKER_REGISTERED", worker_id=w.worker_id, backend=w.backend,
                  data=f"{w.data_host}:{w.data_port}")
        return {"worker_id": w.worker_id, "heartbeat_interval_s": max(1.0, c.registry.heartbeat_timeout_s / 4)}

    @app.post("/workers/{worker_id}/heartbeat", dependencies=[Depends(auth)])
    def heartbeat(worker_id: str, req: HeartbeatRequest):
        if not c.registry.heartbeat(worker_id, req.memory):
            raise HTTPException(404, "unknown worker; re-register")
        return {"ok": True}

    @app.get("/workers/{worker_id}/commands", dependencies=[Depends(auth)])
    def commands(worker_id: str, timeout: float = 20.0):
        worker_or_404(worker_id)
        c.registry.heartbeat(worker_id)
        return {"commands": c.registry.poll(worker_id, min(max(timeout, 0.0), 30.0))}

    @app.post("/workers/{worker_id}/events", dependencies=[Depends(auth)])
    def events(worker_id: str, ev: EventRequest):
        worker_or_404(worker_id)
        c.handle_event(worker_id, ev)
        return {"ok": True}

    @app.get("/cluster/status", dependencies=[Depends(auth)])
    def status():
        return {"workers": [w.public() for w in c.registry.all()],
                "jobs": [{"job_id": j.job_id, "status": j.status.value} for j in c.jobs.values()]}

    @app.post("/cluster/benchmark", dependencies=[Depends(auth)])
    def benchmark(req: BenchmarkRequest):
        return c.start_benchmark(req)

    @app.get("/cluster/benchmark", dependencies=[Depends(auth)])
    def benchmark_status():
        return c.benchmark

    @app.post("/plan", dependencies=[Depends(auth)])
    def plan(req: JobRequest):
        try:
            cfg, p = c.plan(req.config)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        names = [cfg.build_model_spec().layer_name(i) for i in range(cfg.model_num_layers())]
        return {"plan": p.to_dict(), "text": p.format(names)}

    @app.post("/jobs", dependencies=[Depends(auth)])
    def start(req: JobRequest):
        try:
            job = c.start_job(req.config)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        names_spec = parse_config(req.config).build_model_spec()
        text = (Path(job.run_dir) / "plan.txt").read_text()
        return {**job.public(), "plan_text": text, "num_layers": names_spec.num_layers}

    @app.get("/jobs", dependencies=[Depends(auth)])
    def jobs():
        return {"jobs": [j.public() for j in c.jobs.values()]}

    @app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str):
        if job_id not in c.jobs:
            raise HTTPException(404, "unknown job")
        return c.jobs[job_id].public()

    @app.post("/jobs/{job_id}/stop", dependencies=[Depends(auth)])
    def stop(job_id: str):
        if job_id not in c.jobs:
            raise HTTPException(404, "unknown job")
        return c.stop_job(job_id).public()

    return app


def run_coordinator(host: str = "0.0.0.0", port: int = 8080, *, token: str, runs_dir: str = "runs",
                    heartbeat_timeout_s: float = 15.0) -> None:
    import uvicorn

    coordinator = Coordinator(token, runs_dir, heartbeat_timeout_s)
    print(f"MeshTrain coordinator listening on http://{host}:{port}  (runs -> {runs_dir})", flush=True)
    uvicorn.run(create_app(coordinator), host=host, port=port, log_level="warning")

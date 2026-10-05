"""Coordinator state records (in memory; V1 has no persistence)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class WorkerStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"


class JobStatus(str, enum.Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class WorkerRecord:
    worker_id: str
    name: str
    hardware: dict
    backend: str                 # device the worker will run stages on
    device_info: dict            # adapter.describe()
    data_host: str
    data_port: int
    status: WorkerStatus = WorkerStatus.ONLINE
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    memory: dict = field(default_factory=dict)
    benchmark: dict | None = None
    current_job: str | None = None

    def public(self) -> dict:
        return {
            "worker_id": self.worker_id, "name": self.name, "hostname": self.hardware.get("hostname"),
            "platform": self.hardware.get("platform"), "cpu_count": self.hardware.get("cpu_count"),
            "ram_total": self.hardware.get("ram_total"), "accelerators": self.hardware.get("accelerators", []),
            "capabilities": self.hardware.get("capabilities", []), "backend": self.backend,
            "device": self.device_info, "data_address": f"{self.data_host}:{self.data_port}",
            "status": self.status.value, "last_heartbeat_age_s": round(time.time() - self.last_heartbeat, 1),
            "memory": self.memory, "benchmark": self.benchmark, "current_job": self.current_job,
        }


@dataclass
class JobRecord:
    job_id: str
    config: dict
    plan: dict
    status: JobStatus = JobStatus.PLANNED
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stage_workers: list[str] = field(default_factory=list)
    stages_done: dict[int, dict] = field(default_factory=dict)
    losses: list[tuple[int, float]] = field(default_factory=list)
    last_metrics: dict[int, dict] = field(default_factory=dict)
    run_dir: str = ""
    summary: dict | None = None
    attempt: int = 0                                      # placement attempt (startup replanning)
    attempts: list[dict] = field(default_factory=list)
    budget_overrides: dict[str, int] = field(default_factory=dict)
    memory_reports: dict[str, dict] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "name": self.config.get("job", {}).get("name"), "status": self.status.value,
            "error": self.error, "plan": self.plan, "stage_workers": self.stage_workers,
            "losses": self.losses, "last_metrics": self.last_metrics,
            "stages_done": self.stages_done, "run_dir": self.run_dir, "summary": self.summary,
            "attempt": self.attempt, "attempts": self.attempts, "memory_reports": self.memory_reports,
            "created_at": self.created_at, "finished_at": self.finished_at,
        }

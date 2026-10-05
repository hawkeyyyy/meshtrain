"""Worker registry, liveness tracking and per-worker command queues."""

from __future__ import annotations

import queue
import re
import threading
import time
import uuid
from typing import Callable

from meshtrain.coordinator.state import WorkerRecord, WorkerStatus


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-")[:48] or "worker"


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_s: float = 15.0):
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._workers: dict[str, WorkerRecord] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.RLock()
        self.on_offline: list[Callable[[WorkerRecord], None]] = []

    def register(self, *, name: str, hardware: dict, backend: str, device_info: dict, data_host: str,
                 data_port: int) -> WorkerRecord:
        with self._lock:
            base = _slug(name)
            # Re-registration of the same name+address reuses the id (worker restart).
            for w in self._workers.values():
                if w.name == base and (w.data_host, w.data_port) == (data_host, data_port):
                    w.hardware, w.backend, w.device_info = hardware, backend, device_info
                    w.status, w.last_heartbeat, w.current_job = WorkerStatus.ONLINE, time.time(), None
                    return w
            worker_id = base if base not in self._workers else f"{base}-{uuid.uuid4().hex[:4]}"
            rec = WorkerRecord(worker_id, worker_id, hardware, backend, device_info, data_host, data_port)
            self._workers[worker_id] = rec
            self._queues[worker_id] = queue.Queue()
            return rec

    def get(self, worker_id: str) -> WorkerRecord | None:
        with self._lock:
            return self._workers.get(worker_id)

    def all(self) -> list[WorkerRecord]:
        with self._lock:
            return list(self._workers.values())

    def online(self) -> list[WorkerRecord]:
        return [w for w in self.all() if w.status != WorkerStatus.OFFLINE]

    def heartbeat(self, worker_id: str, memory: dict | None = None) -> bool:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is None:
                return False
            w.last_heartbeat = time.time()
            if memory:
                w.memory = memory
            if w.status == WorkerStatus.OFFLINE:
                w.status = WorkerStatus.ONLINE
            return True

    def check_liveness(self) -> list[WorkerRecord]:
        """Mark workers whose heartbeat is stale as OFFLINE; returns newly offline workers."""
        now = time.time()
        newly = []
        with self._lock:
            for w in self._workers.values():
                if w.status != WorkerStatus.OFFLINE and now - w.last_heartbeat > self.heartbeat_timeout_s:
                    w.status = WorkerStatus.OFFLINE
                    newly.append(w)
        for w in newly:
            for cb in self.on_offline:
                cb(w)
        return newly

    # -- commands ---------------------------------------------------------
    def send(self, worker_id: str, command: dict) -> None:
        with self._lock:
            q = self._queues[worker_id]
        q.put(command)

    def poll(self, worker_id: str, timeout: float) -> list[dict]:
        with self._lock:
            q = self._queues.get(worker_id)
        if q is None:
            raise KeyError(worker_id)
        out = []
        try:
            out.append(q.get(timeout=timeout))
            while True:
                out.append(q.get_nowait())
        except queue.Empty:
            pass
        return out

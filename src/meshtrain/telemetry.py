"""Structured event logging and metrics files.

Every event line carries timestamp, worker, job, step, microbatch and event
name, e.g.::

    [17:12:04.512] worker=rtx8 job=j1 step=12 mb=3 FORWARD_COMPLETE output=16.2MB compute=83.0ms

Events can also be mirrored to a JSONL file for later analysis.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1000
    return f"{n:.1f}TB"


def fmt_value(key: str, value: Any) -> str:
    if isinstance(value, float):
        if key.endswith("_s"):
            return f"{value * 1000:.1f}ms"
        return f"{value:.4g}"
    if isinstance(value, int) and (key.endswith("bytes") or key in ("output", "input")):
        return fmt_bytes(value)
    return str(value)


class EventLogger:
    def __init__(self, worker: str, job: str = "-", *, jsonl_path: str | Path | None = None,
                 stream=None, verbose: bool = True):
        self.worker = worker
        self.job = job
        self.stream = stream if stream is not None else sys.stdout
        self.verbose = verbose and os.environ.get("MESHTRAIN_QUIET") != "1"
        self._lock = threading.Lock()
        self._file = None
        if jsonl_path is not None:
            Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(jsonl_path, "a")

    def log(self, event: str, step: int | None = None, microbatch: int | None = None, *,
            echo: bool = True, **fields: Any) -> dict:
        now = time.time()
        record = {"timestamp": now, "worker": self.worker, "job": self.job, "step": step,
                  "microbatch": microbatch, "event": event, **fields}
        with self._lock:
            if self.verbose and echo:
                ts = datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3]
                parts = [f"[{ts}]", f"worker={self.worker}", f"job={self.job}"]
                if step is not None:
                    parts.append(f"step={step}")
                if microbatch is not None:
                    parts.append(f"mb={microbatch}")
                parts.append(event)
                parts += [f"{k}={fmt_value(k, v)}" for k, v in fields.items()]
                print(" ".join(parts), file=self.stream, flush=True)
            if self._file is not None:
                self._file.write(json.dumps(record, default=str) + "\n")
                self._file.flush()
        return record

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class MetricsWriter:
    """Append-only ``runs/<run-id>/metrics.jsonl``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid() % 10000:04d}"

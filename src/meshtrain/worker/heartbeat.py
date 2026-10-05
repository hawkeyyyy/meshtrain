"""Background heartbeat thread."""

from __future__ import annotations

import threading
from typing import Callable

from meshtrain.networking.control import ControlClient, ControlError


class Heartbeat:
    def __init__(self, client: ControlClient, worker_id_fn: Callable[[], str], memory_fn: Callable[[], dict],
                 interval_s: float = 3.0, on_unknown: Callable[[], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None):
        self.client = client
        self.worker_id_fn = worker_id_fn
        self.memory_fn = memory_fn
        self.interval_s = interval_s
        self.on_unknown = on_unknown
        self.on_error = on_error
        self.failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")

    def start(self) -> "Heartbeat":
        self._thread.start()
        return self

    def beat(self) -> None:
        try:
            self.client.heartbeat(self.worker_id_fn(), self.memory_fn())
            self.failures = 0
        except ControlError as exc:
            self.failures += 1
            if "404" in str(exc) and self.on_unknown:
                self.on_unknown()  # coordinator restarted: re-register
            elif self.on_error:
                self.on_error(exc)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.beat()

    def stop(self) -> None:
        self._stop.set()

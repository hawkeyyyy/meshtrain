"""HTTP client for the coordinator's control plane (used by workers and the CLI)."""

from __future__ import annotations

import httpx

DEFAULT_PORT = 8080


def normalize_address(address: str) -> str:
    if address.startswith(("http://", "https://")):
        return address.rstrip("/")
    if ":" not in address:
        address = f"{address}:{DEFAULT_PORT}"
    return f"http://{address}"


class ControlError(RuntimeError):
    pass


class ControlClient:
    def __init__(self, address: str, token: str, timeout: float = 10.0, transport=None):
        self.base = normalize_address(address)
        kwargs = {"transport": transport} if transport is not None else {}
        self.http = httpx.Client(base_url=self.base, timeout=timeout,
                                 headers={"X-MeshTrain-Token": token}, **kwargs)

    def _call(self, method: str, path: str, *, timeout: float | None = None, **kw) -> dict:
        try:
            r = self.http.request(method, path, timeout=timeout if timeout is not None else self.http.timeout, **kw)
        except httpx.HTTPError as exc:
            raise ControlError(f"coordinator {self.base} unreachable: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ControlError(f"{method} {path} -> {r.status_code}: {detail}")
        return r.json()

    # worker side
    def register(self, payload: dict) -> dict:
        return self._call("POST", "/workers/register", json=payload)

    def heartbeat(self, worker_id: str, memory: dict) -> dict:
        return self._call("POST", f"/workers/{worker_id}/heartbeat", json={"memory": memory})

    def poll_commands(self, worker_id: str, wait_s: float = 20.0) -> list[dict]:
        return self._call("GET", f"/workers/{worker_id}/commands", params={"timeout": wait_s},
                          timeout=wait_s + 10)["commands"]

    def event(self, worker_id: str, type_: str, data: dict) -> None:
        self._call("POST", f"/workers/{worker_id}/events", json={"type": type_, "data": data})

    # user side
    def status(self) -> dict:
        return self._call("GET", "/cluster/status")

    def start_benchmark(self, **kw) -> dict:
        return self._call("POST", "/cluster/benchmark", json=kw)

    def benchmark(self) -> dict:
        return self._call("GET", "/cluster/benchmark")

    def plan(self, config: dict) -> dict:
        return self._call("POST", "/plan", json={"config": config}, timeout=120)

    def start_job(self, config: dict) -> dict:
        return self._call("POST", "/jobs", json={"config": config}, timeout=120)

    def job(self, job_id: str) -> dict:
        return self._call("GET", f"/jobs/{job_id}")

    def jobs(self) -> dict:
        return self._call("GET", "/jobs")

    def stop_job(self, job_id: str) -> dict:
        return self._call("POST", f"/jobs/{job_id}/stop")

    def close(self) -> None:
        self.http.close()

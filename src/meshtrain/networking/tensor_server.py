"""Remote tensor memory: a RAM-only tensor service and its client (V2.5).

A worker that contributes RAM runs a ``TensorServer``. Clients (stages that
compute) store tensors there by stable TensorStore id and fetch them back
before use. The server never computes and never interprets payloads: it
keeps raw bytes plus metadata (dtype, shape, version, crc32) per
``(job_id, tensor_id)``.

Wire format: the existing data-plane framing (networking/protocol.py) on a
TCP connection opened with a HELLO ``{"command_kind": "TENSOR_STORE"}`` and
the cluster token. Requests are ``MessageType.CONTROL`` packets whose
``meta["op"]`` is one of::

    RESERVE  {job, bytes}                        reserve RAM for a job (fails if unavailable)
    RELEASE  {job}                               drop a job's tensors and reservation
    PUT      {job, tensor_id, old_version, new_version, crc32?} + tensor payload
    GET      {job, tensor_id, version?}         -> tensor payload, {version, crc32}
    META     {job, tensor_id}                    -> {version, dtype, shape, bytes, crc32}
    DELETE   {job, tensor_id}
    HEARTBEAT{job}                               keeps a job's lease alive
    STATS    {}                                  -> server-wide accounting
    PING     {} (+ optional payload)             -> echo, for RTT / bandwidth probes

Replies are ``ACK`` packets (``meta["ok"] = True`` plus fields) or ``ERROR``
packets (``meta["error"]``, ``meta["code"]``).

Consistency (hard invariant: one authoritative copy per logical tensor):
``PUT`` is accepted only if the stored version equals ``old_version``
(``-1`` = must not exist) and then atomically advances it to
``new_version``. A stale writer gets ``VERSION_CONFLICT`` and nothing
changes. ``GET`` with ``version`` fails with ``VERSION_MISMATCH`` if the
server holds a different version.

Capacity: ``budget`` bytes in total for all jobs. ``RESERVE`` must succeed
before any ``PUT`` of that job, and a job's stored bytes can never exceed
its reservation. A job whose client is silent for ``lease_s`` seconds is
released automatically, so a crashed client cannot leak remote RAM.

Trusted private networks only (V2.5): the shared cluster token
authenticates, sizes and ids are validated, payloads are opaque bytes and
nothing is ever executed or unpickled. There is no encryption.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import zlib
from concurrent.futures import Future
from dataclasses import dataclass

from meshtrain.networking.protocol import DEFAULT_MAX_PAYLOAD_BYTES
from meshtrain.networking.tcp import TCPListener, TCPTransport, connect
from meshtrain.networking.transport import TransportClosed, TransportTimeout
from meshtrain.runtime.tensor_packet import NAME_TO_DTYPE, MessageType, TensorPacket, dtype_itemsize

MAX_ID = 200
_OPS = {"RESERVE", "RELEASE", "PUT", "GET", "META", "DELETE", "HEARTBEAT", "STATS", "PING"}


class RemoteMemoryError(RuntimeError):
    """A remote tensor operation failed (``code`` says why)."""

    def __init__(self, message: str, code: str = "ERROR", worker: str = "", tensor_id: str = ""):
        super().__init__(message)
        self.code, self.worker, self.tensor_id = code, worker, tensor_id


class VersionConflict(RemoteMemoryError):
    pass


def crc32(data) -> int:
    return zlib.crc32(memoryview(data).cast("B")) & 0xFFFFFFFF


@dataclass
class _Entry:
    payload: bytearray
    dtype: str
    shape: tuple[int, ...]
    version: int
    crc: int | None


class _Job:
    def __init__(self, reserved: int):
        self.reserved = reserved
        self.used = 0
        self.tensors: dict[str, _Entry] = {}
        self.last_seen = time.monotonic()
        self.lock = threading.Lock()


# ---------------------------------------------------------------------------- server
class TensorServer:
    """RAM tensor store. ``handle(link)`` serves one authenticated connection."""

    def __init__(self, budget_bytes: int, *, name: str = "tensor-server", lease_s: float = 120.0,
                 max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES, reserve_system_bytes: int = 0):
        self.budget = int(budget_bytes)
        # RAM the OS and other programs keep: a reservation is refused if it would leave less free.
        self.reserve_system = int(reserve_system_bytes)
        self.name = name
        self.lease_s = lease_s
        self.max_payload_bytes = max_payload_bytes
        self.jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self.counters = {"puts": 0, "gets": 0, "bytes_in": 0, "bytes_out": 0, "conflicts": 0, "rejected": 0,
                         "expired_jobs": 0}
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="tensor-lease-reaper")
        self._reaper.start()

    # -- accounting ----------------------------------------------------------
    @property
    def reserved(self) -> int:
        with self._lock:
            return sum(j.reserved for j in self.jobs.values())

    @property
    def used(self) -> int:
        with self._lock:
            return sum(j.used for j in self.jobs.values())

    def advertisement(self) -> dict:
        """Flat numbers for worker heartbeats (``meshtrain remote status``)."""
        import psutil

        vm = psutil.virtual_memory()
        st = self.stats()
        return {"remote_ram_budget": self.budget, "remote_ram_reserved": st["reserved"],
                "remote_ram_used": st["used"], "remote_ram_jobs": len(st["jobs"]),
                "ram_total": int(vm.total), "ram_available": int(vm.available),
                "remote_ram_reserve_system": self.reserve_system}

    def stats(self) -> dict:
        with self._lock:
            return {"name": self.name, "budget": self.budget,
                    "reserved": sum(j.reserved for j in self.jobs.values()),
                    "used": sum(j.used for j in self.jobs.values()),
                    "jobs": {k: {"reserved": j.reserved, "used": j.used, "tensors": len(j.tensors)}
                             for k, j in self.jobs.items()},
                    **self.counters}

    def _reap(self) -> None:
        while not self._stop.wait(min(5.0, self.lease_s / 4)):
            now = time.monotonic()
            with self._lock:
                dead = [k for k, j in self.jobs.items() if now - j.last_seen > self.lease_s]
                for k in dead:
                    del self.jobs[k]
                    self.counters["expired_jobs"] += 1

    def close(self) -> None:
        self._stop.set()

    # -- request handling ------------------------------------------------------
    def _job(self, job_id: str) -> _Job:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise RemoteMemoryError(f"job {job_id!r} has no reservation on {self.name}", "NO_RESERVATION")
        job.last_seen = time.monotonic()
        return job

    def process(self, packet: TensorPacket) -> TensorPacket:
        """Execute one request packet and return the reply packet."""
        meta = packet.meta
        op = meta.get("op")
        job_id = str(meta.get("job", ""))
        tid = str(meta.get("tensor_id", ""))
        try:
            if op not in _OPS:
                raise RemoteMemoryError(f"unknown op {op!r}", "BAD_REQUEST")
            if len(job_id) > MAX_ID or len(tid) > MAX_ID:
                raise RemoteMemoryError("id too long", "BAD_REQUEST")
            if op == "PING":
                return _ack(payload=packet.payload, dtype=packet.dtype, shape=packet.shape, server=self.name)
            if op == "STATS":
                st = self.stats()
                return _ack(budget=st["budget"], reserved=st["reserved"], used=st["used"], jobs=len(st["jobs"]),
                            puts=st["puts"], gets=st["gets"], server=self.name)
            if not job_id:
                raise RemoteMemoryError("request without job id", "BAD_REQUEST")
            if op == "RESERVE":
                want = int(meta.get("bytes", 0))
                if want < 0:
                    raise RemoteMemoryError("negative reservation", "BAD_REQUEST")
                with self._lock:
                    others = sum(j.reserved for k, j in self.jobs.items() if k != job_id)
                    job = self.jobs.get(job_id)
                    if others + want > self.budget:
                        self.counters["rejected"] += 1
                        raise RemoteMemoryError(
                            f"{self.name}: cannot reserve {want / 1024**2:.0f} MB for {job_id}: budget "
                            f"{self.budget / 1024**2:.0f} MB, {others / 1024**2:.0f} MB reserved by other jobs",
                            "INSUFFICIENT_MEMORY")
                    grow = want - (job.reserved if job is not None else 0)
                    if self.reserve_system and grow > 0:
                        import psutil

                        avail = psutil.virtual_memory().available
                        if avail - grow < self.reserve_system:
                            self.counters["rejected"] += 1
                            raise RemoteMemoryError(
                                f"{self.name}: reserving {grow / 1024**2:.0f} MB more would leave "
                                f"{(avail - grow) / 1024**3:.2f} GB free, below the "
                                f"{self.reserve_system / 1024**3:.1f} GB kept for the system", "INSUFFICIENT_MEMORY")
                    if job is None:
                        self.jobs[job_id] = _Job(want)
                    else:
                        if want < job.used:
                            raise RemoteMemoryError("reservation below bytes already stored", "BAD_REQUEST")
                        job.reserved = want
                return _ack(reserved=want, budget=self.budget)
            if op == "RELEASE":
                with self._lock:
                    job = self.jobs.pop(job_id, None)
                return _ack(released=job.used if job else 0)
            job = self._job(job_id)
            if op == "HEARTBEAT":
                return _ack(used=job.used, reserved=job.reserved)
            if not tid:
                raise RemoteMemoryError("request without tensor id", "BAD_REQUEST")
            if op == "PUT":
                return self._put(job, job_id, tid, packet)
            with job.lock:
                entry = job.tensors.get(tid)
                if entry is None:
                    raise RemoteMemoryError(f"{tid} is not stored on {self.name}", "NOT_FOUND", self.name, tid)
                if op == "GET":
                    want = meta.get("version")
                    if want is not None and int(want) != entry.version:
                        raise RemoteMemoryError(f"{tid}: requested version {want}, stored {entry.version}",
                                                "VERSION_MISMATCH", self.name, tid)
                    self.counters["gets"] += 1
                    self.counters["bytes_out"] += len(entry.payload)
                    return _ack(payload=entry.payload, dtype=entry.dtype, shape=entry.shape, version=entry.version,
                                crc32=entry.crc)
                if op == "META":
                    return _ack(version=entry.version, tensor_dtype=entry.dtype,
                                tensor_shape=",".join(map(str, entry.shape)), bytes=len(entry.payload), crc32=entry.crc)
                if op == "DELETE":
                    del job.tensors[tid]
                    job.used -= len(entry.payload)
                    return _ack(freed=len(entry.payload))
            raise RemoteMemoryError(f"unknown op {op!r}", "BAD_REQUEST")
        except VersionConflict as exc:
            self.counters["conflicts"] += 1
            return _error(str(exc), exc.code)
        except RemoteMemoryError as exc:
            return _error(str(exc), exc.code)
        except (TypeError, ValueError) as exc:
            return _error(f"bad request: {exc}", "BAD_REQUEST")

    def _put(self, job: _Job, job_id: str, tid: str, packet: TensorPacket) -> TensorPacket:
        meta = packet.meta
        if packet.dtype is None or packet.dtype not in NAME_TO_DTYPE:
            raise RemoteMemoryError("PUT without a tensor payload", "BAD_REQUEST")
        nbytes = math.prod(packet.shape) * dtype_itemsize(packet.dtype)
        if nbytes != len(packet.payload):
            raise RemoteMemoryError("payload size does not match shape/dtype", "BAD_REQUEST")
        crc = meta.get("crc32")
        if crc is not None and crc32(packet.payload) != int(crc):
            raise RemoteMemoryError(f"{tid}: checksum mismatch", "CHECKSUM", self.name, tid)
        old, new = int(meta.get("old_version", -1)), int(meta.get("new_version", 0))
        with job.lock:
            entry = job.tensors.get(tid)
            stored = entry.version if entry is not None else -1
            if stored != old:
                raise VersionConflict(f"{tid}: write expects version {old}, {self.name} holds {stored}",
                                      "VERSION_CONFLICT", self.name, tid)
            if new <= old:
                raise RemoteMemoryError(f"{tid}: new version {new} must exceed {old}", "BAD_REQUEST")
            delta = nbytes - (len(entry.payload) if entry is not None else 0)
            if job.used + delta > job.reserved:
                raise RemoteMemoryError(
                    f"{tid}: job {job_id} would store {(job.used + delta) / 1024**2:.1f} MB, reservation "
                    f"{job.reserved / 1024**2:.1f} MB", "RESERVATION_EXCEEDED", self.name, tid)
            payload = packet.payload if isinstance(packet.payload, bytearray) else bytearray(packet.payload)
            job.tensors[tid] = _Entry(payload, packet.dtype, tuple(packet.shape), new,
                                      int(crc) if crc is not None else None)
            job.used += delta
        self.counters["puts"] += 1
        self.counters["bytes_in"] += nbytes
        return _ack(version=new)

    def handle(self, link: TCPTransport, stop_event: threading.Event | None = None) -> None:
        """Serve requests on one connection until it closes."""
        try:
            while not self._stop.is_set() and not (stop_event is not None and stop_event.is_set()):
                try:
                    packet = link.recv_packet(timeout=1.0)
                except TransportTimeout:
                    continue
                if packet.message_type != MessageType.CONTROL:
                    link.send_packet(_error("expected a CONTROL request", "BAD_REQUEST"))
                    continue
                reply = self.process(packet)
                reply.tensor_id = packet.tensor_id  # correlate request/response
                link.send_packet(reply)
        except (TransportClosed, OSError):
            pass
        finally:
            link.close()


def _ack(payload=b"", dtype=None, shape=(), **meta) -> TensorPacket:
    return TensorPacket(MessageType.ACK, payload=payload, dtype=dtype, shape=tuple(shape),
                        meta={"ok": True, **{k: v for k, v in meta.items() if v is not None}})


def _error(message: str, code: str) -> TensorPacket:
    return TensorPacket(MessageType.ERROR, meta={"ok": False, "error": message[:240], "code": code})


def serve_tensor_store(host: str, port: int, budget_bytes: int, *, token: str | None, name: str = "tensor-server",
                       lease_s: float = 120.0, stop_event: threading.Event | None = None,
                       reserve_system_bytes: int = 0, on_ready=None) -> None:
    """Standalone server loop (``meshtrain tensor-server``)."""
    listener = TCPListener(host, port, token=token)
    server = TensorServer(budget_bytes, name=name, lease_s=lease_s, reserve_system_bytes=reserve_system_bytes)
    if on_ready is not None:
        on_ready(server, listener.port)
    try:
        while not (stop_event is not None and stop_event.is_set()):
            try:
                link, hello = listener.accept(timeout=0.5)
            except TransportTimeout:
                continue
            except Exception:
                continue  # bad hello / token
            if hello.get("command_kind") != "TENSOR_STORE":
                link.close()
                continue
            threading.Thread(target=server.handle, args=(link, stop_event), daemon=True).start()
    finally:
        listener.close()
        server.close()


# ---------------------------------------------------------------------------- client
@dataclass
class TransferTiming:
    queue_wait_s: float = 0.0     # waiting for a free connection (in-flight limit)
    network_s: float = 0.0        # request sent -> reply fully received
    bytes: int = 0


class RemoteTensorClient:
    """Client for one remote RAM worker.

    ``max_inflight`` connections, each serving one request at a time, bound
    the number of outstanding GET/PUT requests (back-pressure: callers block
    in ``_acquire`` instead of buffering unboundedly). Separate pools for
    fetches and writebacks keep a slow writeback from blocking a fetch.
    """

    def __init__(self, host: str, port: int, *, token: str | None, job_id: str, worker: str = "",
                 max_inflight_fetches: int = 2, max_inflight_writebacks: int = 2, checksum: str = "none",
                 timeout_s: float = 120.0, link_wrapper=None):
        self.host, self.port, self.token, self.job_id = host, port, token, job_id
        self.worker = worker or f"{host}:{port}"
        self.checksum = checksum
        self.timeout_s = timeout_s
        self.link_wrapper = link_wrapper   # e.g. EmulatedLink for bandwidth experiments
        self._pools = {"fetch": queue.Queue(), "writeback": queue.Queue(), "control": queue.Queue()}
        self._sizes = {"fetch": max_inflight_fetches, "writeback": max_inflight_writebacks, "control": 1}
        self._made = {k: 0 for k in self._pools}
        self._lock = threading.Lock()
        self.closed = False
        self.counters = {"gets": 0, "puts": 0, "bytes_read": 0, "bytes_written": 0, "network_s": 0.0,
                         "queue_wait_s": 0.0}
        self._hb_stop = threading.Event()

    def _connect(self) -> TCPTransport:
        hello = {"command_kind": "TENSOR_STORE", "job_id": self.job_id}
        if self.token is not None:
            hello["token"] = self.token
        link = connect(self.host, self.port, timeout=min(self.timeout_s, 30.0), hello=hello,
                       frame_timeout_s=self.timeout_s)
        return self.link_wrapper(link) if self.link_wrapper is not None else link

    def _acquire(self, pool: str):
        q = self._pools[pool]
        with self._lock:
            if q.empty() and self._made[pool] < self._sizes[pool]:
                self._made[pool] += 1
                try:
                    return self._connect()
                except Exception as exc:
                    self._made[pool] -= 1
                    raise RemoteMemoryError(f"cannot reach remote memory worker {self.worker}: {exc}",
                                            "UNREACHABLE", self.worker) from exc
        try:
            return q.get(timeout=self.timeout_s)
        except queue.Empty:
            raise RemoteMemoryError(f"no free connection to {self.worker} within {self.timeout_s}s "
                                    f"(in-flight limit {self._sizes[pool]})", "TIMEOUT", self.worker) from None

    def _release(self, pool: str, link, broken: bool) -> None:
        if broken:
            try:
                link.close()
            except Exception:
                pass
            with self._lock:
                self._made[pool] -= 1
        else:
            self._pools[pool].put(link)

    def request(self, op: str, *, pool: str = "control", payload=b"", dtype=None, shape=(), into=None,
                timing: TransferTiming | None = None, **meta) -> TensorPacket:
        if self.closed:
            raise RemoteMemoryError(f"client for {self.worker} is closed", "CLOSED", self.worker)
        t0 = time.perf_counter()
        link = self._acquire(pool)
        t1 = time.perf_counter()
        broken = False
        try:
            if into is not None:   # receive the reply payload straight into a (pinned) staging buffer
                link.payload_allocator = lambda mtype, dt, shp: into if tuple(shp) == tuple(into.shape) else None
            link.send_packet(TensorPacket(MessageType.CONTROL, job_id=self.job_id, payload=payload, dtype=dtype,
                                          shape=tuple(shape), meta={"op": op, "job": self.job_id, **meta}))
            reply = link.recv_packet(timeout=self.timeout_s)
        except (TransportClosed, TransportTimeout, OSError) as exc:
            broken = True
            raise RemoteMemoryError(f"remote memory worker {self.worker} failed during {op} "
                                    f"{meta.get('tensor_id', '')}: {exc}", "DISCONNECTED", self.worker,
                                    str(meta.get("tensor_id", ""))) from exc
        finally:
            if into is not None and not broken:
                link.payload_allocator = None
            self._release(pool, link, broken)
        t2 = time.perf_counter()
        self.counters["queue_wait_s"] += t1 - t0
        self.counters["network_s"] += t2 - t1
        if timing is not None:
            timing.queue_wait_s += t1 - t0
            timing.network_s += t2 - t1
        if reply.message_type == MessageType.ERROR:
            code = str(reply.meta.get("code", "ERROR"))
            cls = VersionConflict if code == "VERSION_CONFLICT" else RemoteMemoryError
            raise cls(f"{self.worker}: {reply.meta.get('error')}", code, self.worker, str(meta.get("tensor_id", "")))
        return reply

    # -- RemoteTensorStore operations ---------------------------------------
    def reserve(self, nbytes: int) -> dict:
        return self.request("RESERVE", bytes=int(nbytes)).meta

    def release(self) -> dict:
        return self.request("RELEASE").meta

    def heartbeat(self) -> dict:
        return self.request("HEARTBEAT").meta

    def stats(self) -> dict:
        return self.request("STATS").meta

    def put(self, tensor_id: str, tensor, *, old_version: int, new_version: int,
            timing: TransferTiming | None = None) -> int:
        """Write a CPU tensor; returns the committed version (raises VersionConflict on a stale write)."""
        from meshtrain.runtime.serialization import tensor_to_bytes

        payload, dtype, shape = tensor_to_bytes(tensor.detach().contiguous(), copy=False)
        meta = {"tensor_id": tensor_id, "old_version": int(old_version), "new_version": int(new_version)}
        if self.checksum == "crc32":
            meta["crc32"] = crc32(payload)
        reply = self.request("PUT", pool="writeback", payload=payload, dtype=dtype, shape=shape, timing=timing,
                             **meta)
        self.counters["puts"] += 1
        self.counters["bytes_written"] += len(payload)
        if timing is not None:
            timing.bytes += len(payload)
        return int(reply.meta["version"])

    def get(self, tensor_id: str, into, *, version: int | None = None,
            timing: TransferTiming | None = None) -> int:
        """Fetch into the CPU tensor ``into`` (shape/dtype must match); returns the version."""
        import torch

        from meshtrain.runtime.serialization import bytes_to_tensor

        meta = {"tensor_id": tensor_id}
        if version is not None:
            meta["version"] = int(version)
        reply = self.request("GET", pool="fetch", into=into, timing=timing, **meta)
        if getattr(reply, "recv_buffer", None) is None:   # did not land in place: copy (validated decode)
            src = bytes_to_tensor(reply.payload, reply.dtype, reply.shape)
            if tuple(src.shape) != tuple(into.shape) or src.dtype != into.dtype:
                raise RemoteMemoryError(f"{tensor_id}: remote {tuple(src.shape)} {src.dtype} vs local "
                                        f"{tuple(into.shape)} {into.dtype}", "SHAPE_MISMATCH", self.worker, tensor_id)
            with torch.no_grad():
                into.copy_(src)
        crc = reply.meta.get("crc32")
        if self.checksum == "crc32" and crc is not None:
            from meshtrain.runtime.serialization import tensor_to_bytes

            got = crc32(tensor_to_bytes(into, copy=False)[0])
            if got != int(crc):
                raise RemoteMemoryError(f"{tensor_id}: checksum mismatch after fetch from {self.worker}",
                                        "CHECKSUM", self.worker, tensor_id)
        nbytes = into.numel() * into.element_size()
        self.counters["gets"] += 1
        self.counters["bytes_read"] += nbytes
        if timing is not None:
            timing.bytes += nbytes
        return int(reply.meta["version"])

    def metadata(self, tensor_id: str) -> dict:
        return self.request("META", tensor_id=tensor_id).meta

    def exists(self, tensor_id: str) -> bool:
        try:
            self.metadata(tensor_id)
            return True
        except RemoteMemoryError as exc:
            if exc.code == "NOT_FOUND":
                return False
            raise

    def delete(self, tensor_id: str) -> None:
        self.request("DELETE", tensor_id=tensor_id)

    def start_heartbeat(self, every_s: float = 10.0) -> None:
        def loop():
            while not self._hb_stop.wait(every_s):
                try:
                    self.heartbeat()
                except Exception:
                    return
        threading.Thread(target=loop, daemon=True, name=f"remote-hb-{self.worker}").start()

    def probe(self, *, pings: int = 5, payload_mb: float = 32.0) -> dict:
        """RTT (small PINGs) and sustained large-tensor bandwidth both ways."""
        import torch

        rtts = []
        for _ in range(pings):
            t0 = time.perf_counter()
            self.request("PING")
            rtts.append(time.perf_counter() - t0)
        n = int(payload_mb * 1024**2 // 4)
        t = torch.zeros(n, dtype=torch.float32)
        from meshtrain.runtime.serialization import tensor_to_bytes

        payload, dtype, shape = tensor_to_bytes(t)
        t0 = time.perf_counter()
        self.request("PING", payload=payload, dtype=dtype, shape=shape)   # echo: upload + download
        rt = time.perf_counter() - t0
        rtt = sorted(rtts)[len(rtts) // 2]
        return {"rtt_s": rtt, "roundtrip_bytes": 2 * len(payload),
                "bandwidth_Bps": 2 * len(payload) / max(rt - rtt, 1e-9)}

    def close(self) -> None:
        self._hb_stop.set()
        self.closed = True
        for q in self._pools.values():
            while not q.empty():
                try:
                    q.get_nowait().close()
                except Exception:
                    pass


def wrap_future(fn, *args, **kwargs) -> Future:
    """Run ``fn`` on a daemon thread; exceptions surface on ``result()``."""
    fut: Future = Future()

    def run():
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return fut


def start_local_server(budget_bytes: int, *, token: str | None = None, lease_s: float = 120.0, host: str = "127.0.0.1",
                       port: int = 0, name: str = "local-ram"):
    """Tensor service on a background thread (tests, loopback experiments). Returns (server, port, stop)."""
    listener = TCPListener(host, port, token=token)
    server = TensorServer(budget_bytes, name=name, lease_s=lease_s)
    stop = threading.Event()

    def loop():
        try:
            while not stop.is_set():
                try:
                    link, hello = listener.accept(timeout=0.2)
                except TransportTimeout:
                    continue
                except Exception:
                    continue
                if hello.get("command_kind") != "TENSOR_STORE":
                    link.close()
                    continue
                threading.Thread(target=server.handle, args=(link, stop), daemon=True).start()
        finally:
            listener.close()
            server.close()

    threading.Thread(target=loop, daemon=True, name=f"tensor-server-{name}").start()
    return server, listener.port, stop

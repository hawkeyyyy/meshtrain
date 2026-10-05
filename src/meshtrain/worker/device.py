"""Device adapters: the only place that knows about CUDA / MPS specifics.

The runtime asks an adapter to move tensors, synchronize, report memory and
answer capability questions. Adding ROCm / Intel XPU later means adding an
adapter here, not touching the runtime.
"""

from __future__ import annotations

import abc
import os
import platform

import psutil
import torch

from meshtrain.runtime.buffers import BufferPool


class DeviceAdapter(abc.ABC):
    backend: str = "abstract"

    def __init__(self, index: int = 0):
        self.index = index
        self.host_pool = BufferPool(pinned=False, enabled=True)

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    # -- transfer staging ---------------------------------------------------
    # The runtime moves tensors across the explicit network boundary with:
    #   send:    begin_d2h (compute thread) -> finish() (sender thread)
    #   receive: recv_buffer (reader thread) -> begin_h2d (compute thread, early)
    #            -> finish_h2d (just before use) -> after_use (after backward)
    # Defaults are synchronous and correct for any backend; CUDA overrides
    # them with pinned buffers, a transfer stream and events.

    def configure_transfers(self, pinned: bool = True, pool: bool = True) -> None:
        self.host_pool = BufferPool(pinned=pinned and self.supports("pinned_memory"), enabled=pool)

    def recv_buffer(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Host tensor to receive a payload into (from the pool when enabled)."""
        return self.host_pool.acquire(shape, dtype)

    def begin_h2d(self, host: torch.Tensor):
        """Start moving a received host tensor to the device.

        Returns ``(handle, token)``: ``finish_h2d(handle)`` gives the device
        tensor; ``after_use(token)`` is called once the microbatch no longer
        needs it. Default: synchronous copy, host buffer back to the pool.
        """
        t = self.move_tensor(host)
        self.synchronize()
        self.host_pool.release(host)
        return t, None

    def finish_h2d(self, handle) -> torch.Tensor:
        return handle

    def after_use(self, token) -> None:
        if token is not None:
            self.host_pool.release(token)

    def move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.supports_dtype(tensor.dtype):
            raise TypeError(f"{self.backend} does not support dtype {tensor.dtype}")
        return tensor.to(self.device)

    def move_module(self, module: torch.nn.Module) -> torch.nn.Module:
        for p in module.parameters():
            if not self.supports_dtype(p.dtype):
                raise TypeError(f"{self.backend} does not support parameter dtype {p.dtype}")
        return module.to(self.device)

    def synchronize(self) -> None:
        """Block until all queued device work finishes (no-op on CPU)."""

    def sync_compute(self) -> None:
        """Block until the *compute* stream is idle. Used to time forward/
        backward on the host; unlike ``synchronize`` it does not wait for
        transfers running on other streams, so copies keep overlapping."""
        self.synchronize()

    def begin_d2h(self, tensor: torch.Tensor):
        """Start staging ``tensor`` to host memory for sending.

        Called on the compute thread (so it is ordered after the kernels that
        produced ``tensor``). Returns ``finish() -> (cpu_tensor, release)``,
        which may run on another thread; ``release`` (or None) returns any
        staging buffer to its pool once the bytes are sent.

        Default: synchronous copy now, nothing left to do later.
        """
        cpu = tensor.detach().to("cpu")
        self.synchronize()
        return lambda: (cpu, None)

    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def memory_total(self) -> int: ...

    def memory_stats(self) -> dict[str, int | bool]:
        """Bytes: total, allocated (by torch), reserved (by caching allocator)."""
        return {"total": self.memory_total(), "allocated": 0, "reserved": 0, "peak_allocated": 0, "unified": False}

    def reset_peak_memory(self) -> None:
        pass

    def available_memory(self, in_use: int = 0) -> int:
        """Bytes this process could still allocate on the device. ``in_use`` is
        what the caller already holds (used only where the backend cannot
        tell, e.g. emulated CPU budgets)."""
        st = self.memory_stats()
        return max(0, int(st.get("total", 0)) - int(st.get("allocated", 0)))

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return True

    def supports(self, operation: str) -> bool:
        """Capability query. Operation names: 'float64', 'bfloat16', 'float16',
        'autocast', 'pinned_memory', 'nccl'. Unknown operations are assumed
        supported so models fail loudly in torch rather than being silently
        skipped."""
        return {
            "float64": self.supports_dtype(torch.float64),
            "bfloat16": self.supports_dtype(torch.bfloat16),
            "float16": self.supports_dtype(torch.float16),
        }.get(operation, True)

    def describe(self) -> dict:
        return {"backend": self.backend, "name": self.name(), "memory_total": self.memory_total(),
                "unified_memory": bool(self.memory_stats().get("unified"))}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.device})"


class CPUDeviceAdapter(DeviceAdapter):
    backend = "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def begin_d2h(self, tensor: torch.Tensor):
        # Already in host memory: send straight from the tensor (no copy).
        t = tensor.detach()
        return lambda: (t, None)

    def begin_h2d(self, host: torch.Tensor):
        # The received buffer *is* the device tensor; it returns to the pool
        # only after the microbatch's backward (it is saved for backward).
        return host, host

    def name(self) -> str:
        return platform.processor() or platform.machine() or "cpu"

    def memory_total(self) -> int:
        return psutil.virtual_memory().total

    def memory_stats(self):
        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        return {"total": vm.total, "available": vm.available, "allocated": rss, "reserved": rss,
                "peak_allocated": rss, "unified": False}

    def supports(self, operation: str) -> bool:
        if operation in ("pinned_memory", "nccl"):
            return False
        return super().supports(operation)

    def available_memory(self, in_use: int = 0) -> int:
        # Fault injection for tests/experiments: pretend this worker's device has
        # only N GB (the planner still sees the real RAM, so its estimate is
        # "wrong" and runtime validation must catch it).
        emulated = os.environ.get("MESHTRAIN_EMULATE_DEVICE_MEMORY_GB")
        if emulated:
            return max(0, int(float(emulated) * 1024**3) - in_use)
        return int(psutil.virtual_memory().available)


class CUDADeviceAdapter(DeviceAdapter):
    backend = "cuda"

    """CUDA transfers (see docs/performance.md):

    * D2H: the compute thread records an event on the compute stream; the
      sender thread makes a dedicated *transfer stream* wait for that event,
      copies into a pinned pool buffer (non_blocking) and waits only for its
      own copy. Compute kernels queued meanwhile keep running.
    * H2D: as soon as a payload arrives the compute thread enqueues a
      non_blocking copy from the pinned buffer on the transfer stream and
      records an event; just before use the compute stream waits on that
      event (``wait_event``), and the host buffer returns to the pool only
      when the copy's event has completed.
    * ``record_stream`` marks cross-stream tensors so the caching allocator
      never reuses their memory early.
    """

    def __init__(self, index: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        super().__init__(index)
        self.host_pool = BufferPool(pinned=True, enabled=True)
        with torch.cuda.device(index):
            self.transfer_stream = torch.cuda.Stream()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.index)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def sync_compute(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()

    def begin_d2h(self, tensor: torch.Tensor):
        t = tensor.detach()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))

        def finish():
            with torch.cuda.device(self.index):
                buf = self.host_pool.acquire(t.shape, t.dtype)
                with torch.cuda.stream(self.transfer_stream):
                    self.transfer_stream.wait_event(ready)
                    buf.copy_(t, non_blocking=buf.is_pinned())
                    done = torch.cuda.Event()
                    done.record(self.transfer_stream)
                done.synchronize()
            return buf, (lambda: self.host_pool.release(buf))

        return finish

    def begin_h2d(self, host: torch.Tensor):
        if not self.supports_dtype(host.dtype):
            raise TypeError(f"cuda does not support dtype {host.dtype}")
        with torch.cuda.device(self.index), torch.cuda.stream(self.transfer_stream):
            t = host.to(self.device, non_blocking=host.is_pinned())
            ev = torch.cuda.Event()
            ev.record(self.transfer_stream)
        self.host_pool.release(host, ev)  # reusable once the copy has finished
        return (t, ev), None

    def finish_h2d(self, handle) -> torch.Tensor:
        t, ev = handle
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(ev)
        t.record_stream(compute)
        return t

    def name(self) -> str:
        return torch.cuda.get_device_name(self.index)

    def memory_total(self) -> int:
        return torch.cuda.get_device_properties(self.index).total_memory

    def memory_stats(self):
        free, total = torch.cuda.mem_get_info(self.index)
        return {
            "total": total,
            "free": free,
            "allocated": torch.cuda.memory_allocated(self.index),
            "reserved": torch.cuda.memory_reserved(self.index),
            "peak_allocated": torch.cuda.max_memory_allocated(self.index),
            "unified": False,
        }

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.index)

    def available_memory(self, in_use: int = 0) -> int:
        # Free device memory as the driver sees it, plus memory torch has cached
        # but not handed out (reusable without a new driver allocation).
        free, _ = torch.cuda.mem_get_info(self.index)
        cached = torch.cuda.memory_reserved(self.index) - torch.cuda.memory_allocated(self.index)
        return int(free + max(0, cached))

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        if dtype == torch.bfloat16:
            return torch.cuda.is_bf16_supported()
        return True

    def supports(self, operation: str) -> bool:
        if operation in ("pinned_memory", "nccl", "autocast"):
            return True
        return super().supports(operation)


class MPSDeviceAdapter(DeviceAdapter):
    """Apple Silicon. Memory is *unified* with system RAM: ``memory_total``
    reports the recommended working-set size when available, else total RAM."""

    backend = "mps"

    def __init__(self, index: int = 0):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        super().__init__(index)

    @property
    def device(self) -> torch.device:
        return torch.device("mps")

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def name(self) -> str:
        return f"Apple {platform.machine()} (MPS)"

    def memory_total(self) -> int:
        fn = getattr(torch.mps, "recommended_max_memory", None)
        if fn is not None:
            try:
                return int(fn())
            except RuntimeError:
                pass
        return psutil.virtual_memory().total

    def memory_stats(self):
        allocated = int(torch.mps.current_allocated_memory())
        driver = int(getattr(torch.mps, "driver_allocated_memory", lambda: allocated)())
        return {"total": self.memory_total(), "allocated": allocated, "reserved": driver,
                "peak_allocated": driver, "unified": True}

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        # MPS has no float64 kernels.
        return dtype != torch.float64

    def available_memory(self, in_use: int = 0) -> int:
        # Unified memory: the budget is the recommended working set minus what
        # the Metal driver already holds for this process.
        st = self.memory_stats()
        return max(0, int(st["total"]) - int(st["reserved"]))


ADAPTERS: dict[str, type[DeviceAdapter]] = {
    "cpu": CPUDeviceAdapter,
    "cuda": CUDADeviceAdapter,
    "mps": MPSDeviceAdapter,
}


def available_backends() -> list[str]:
    backends = []
    if torch.cuda.is_available():
        backends.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        backends.append("mps")
    backends.append("cpu")
    return backends


def select_device(preferred: str | None = None, index: int = 0) -> DeviceAdapter:
    """Pick an adapter: explicit ``preferred`` ('cuda'|'mps'|'cpu'|'auto'),
    otherwise CUDA, then MPS, then CPU."""
    if preferred in (None, "", "auto"):
        preferred = available_backends()[0]
    preferred = preferred.split(":")[0]
    if preferred not in ADAPTERS:
        raise ValueError(f"unknown backend {preferred!r}; expected one of {sorted(ADAPTERS)}")
    return ADAPTERS[preferred](index)

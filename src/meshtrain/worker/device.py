"""Device adapters: the only place that knows about CUDA / MPS specifics.

The runtime asks an adapter to move tensors, synchronize, report memory and
answer capability questions. Adding ROCm / Intel XPU later means adding an
adapter here, not touching the runtime.
"""

from __future__ import annotations

import abc
import platform

import psutil
import torch


class DeviceAdapter(abc.ABC):
    backend: str = "abstract"

    def __init__(self, index: int = 0):
        self.index = index

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

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
        """Block until queued kernels finish (no-op on CPU)."""

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


class CUDADeviceAdapter(DeviceAdapter):
    backend = "cuda"

    def __init__(self, index: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        super().__init__(index)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.index)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

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

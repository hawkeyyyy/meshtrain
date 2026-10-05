"""Reusable host staging buffers.

``BufferPool`` hands out CPU tensors keyed by ``(shape, dtype, pinned)`` and
takes them back when the runtime is done with them, so steady-state
microbatches stop allocating::

    buf = pool.acquire(shape, dtype)      # reuse a free buffer or allocate
    ...                                   # receive into it / copy from it
    pool.release(buf, event=None)         # back to the free list
                                          # (after ``event`` completes, if given)

* Pinned (page-locked) buffers are used for CUDA staging when requested and
  available; if pinning fails (no CUDA driver, OS limit) the pool falls back
  to pageable memory and records ``pinned_fallbacks``.
* Variable shapes simply create new keys; at most ``max_free_per_key`` idle
  buffers are kept per key and at most ``max_bytes`` idle bytes overall, so
  dynamic shapes cannot grow the pool without bound.
* ``release(buf, event)`` defers reuse until a device event (e.g. the end of
  an async host-to-device copy) has completed.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import torch


class BufferPool:
    def __init__(self, *, pinned: bool = False, enabled: bool = True, max_free_per_key: int = 8,
                 max_bytes: int = 2 * 1024**3):
        self.pinned = pinned
        self.enabled = enabled
        self.max_free_per_key = max_free_per_key
        self.max_bytes = max_bytes
        self._free: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        self._deferred: list[tuple[torch.Tensor, object]] = []
        self._lock = threading.Lock()
        self.allocation_count = 0
        self.reuse_count = 0
        self.pinned_fallbacks = 0
        self.bytes_reserved = 0   # bytes in idle buffers (cached for reuse)
        self.bytes_allocated = 0  # total bytes ever allocated by the pool

    @staticmethod
    def _key(t: torch.Tensor) -> tuple:
        return (tuple(t.shape), t.dtype, t.is_pinned())

    def _alloc(self, shape, dtype) -> torch.Tensor:
        if self.pinned:
            try:
                t = torch.empty(shape, dtype=dtype, pin_memory=True)
            except RuntimeError:
                self.pinned_fallbacks += 1
                t = torch.empty(shape, dtype=dtype)
        else:
            t = torch.empty(shape, dtype=dtype)
        self.allocation_count += 1
        self.bytes_allocated += t.numel() * t.element_size()
        return t

    def _collect_deferred(self) -> None:
        still = []
        for buf, ev in self._deferred:
            if ev is None or ev.query():
                self._put(buf)
            else:
                still.append((buf, ev))
        self._deferred = still

    def _put(self, buf: torch.Tensor) -> None:
        key = self._key(buf)
        nbytes = buf.numel() * buf.element_size()
        if len(self._free[key]) < self.max_free_per_key and self.bytes_reserved + nbytes <= self.max_bytes:
            self._free[key].append(buf)
            self.bytes_reserved += nbytes

    def acquire(self, shape, dtype: torch.dtype) -> torch.Tensor:
        shape = tuple(shape)
        if not self.enabled:
            return self._alloc(shape, dtype)
        with self._lock:
            self._collect_deferred()
            for pinned in ((True, False) if self.pinned else (False,)):
                free = self._free.get((shape, dtype, pinned))
                if free:
                    buf = free.pop()
                    self.bytes_reserved -= buf.numel() * buf.element_size()
                    self.reuse_count += 1
                    return buf
            return self._alloc(shape, dtype)

    def release(self, buf: torch.Tensor, event=None) -> None:
        if not self.enabled or buf is None:
            return
        with self._lock:
            if event is None:
                self._put(buf)
            else:
                self._deferred.append((buf, event))

    def stats(self) -> dict:
        with self._lock:
            return {"allocation_count": self.allocation_count, "reuse_count": self.reuse_count,
                    "bytes_reserved": self.bytes_reserved, "bytes_allocated": self.bytes_allocated,
                    "pinned": self.pinned, "pinned_fallbacks": self.pinned_fallbacks,
                    "deferred": len(self._deferred)}

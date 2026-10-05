"""Hardware detection: what a worker reports when it registers."""

from __future__ import annotations

import os
import platform
import socket

import psutil
import torch

# Accelerators below this much memory (e.g. a 128 MB legacy GPU) are reported
# but marked unusable; such a machine participates as a CPU helper.
MIN_USEFUL_ACCELERATOR_BYTES = 1 * 1024**3


def platform_name() -> str:
    s = platform.system().lower()
    return {"darwin": "macos"}.get(s, s)


def detect_accelerators() -> list[dict]:
    accs = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accs.append({
                "backend": "cuda",
                "index": i,
                "name": props.name,
                "memory_total": int(props.total_memory),
                "compute_capability": f"{props.major}.{props.minor}",
                "unified_memory": False,
                "usable": props.total_memory >= MIN_USEFUL_ACCELERATOR_BYTES,
            })
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        total = psutil.virtual_memory().total
        rec = getattr(torch.mps, "recommended_max_memory", None)
        if rec is not None:
            try:
                total = int(rec())
            except RuntimeError:
                pass
        accs.append({
            "backend": "mps",
            "index": 0,
            "name": f"Apple {platform.machine()}",
            "memory_total": int(total),
            "unified_memory": True,
            "usable": True,
        })
    return accs


def detect_hardware() -> dict:
    accs = detect_accelerators()
    caps = ["cpu", "float32", "float64"]
    for a in accs:
        if a["usable"]:
            caps.append(a["backend"])
    if any(a["backend"] == "cuda" and a["usable"] for a in accs):
        caps += ["float16", "bfloat16" if torch.cuda.is_bf16_supported() else "no-bfloat16"]
    if any(a["backend"] == "mps" for a in accs):
        caps += ["float16", "unified_memory"]
    return {
        "hostname": socket.gethostname(),
        "platform": platform_name(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count() or 1,
        "ram_total": int(psutil.virtual_memory().total),
        "ram_available": int(psutil.virtual_memory().available),
        "accelerators": accs,
        "capabilities": sorted(set(caps)),
    }


def primary_backend(hw: dict) -> str:
    for backend in ("cuda", "mps"):
        if any(a["backend"] == backend and a.get("usable", True) for a in hw.get("accelerators", [])):
            return backend
    return "cpu"

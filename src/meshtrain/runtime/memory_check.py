"""Runtime memory validation at stage startup.

Static estimates are not enough (V1 accepted a split that OOM'd on a 4 GB
GPU). After a stage is materialised on its device, but before training:

1. measure what the stage actually holds and what the device has left;
2. compare with the planner's estimate and record the estimation error;
3. if the rest of the estimated need (gradients, optimizer state,
   activations, buffers) clearly will not fit, fail *now* with a
   ``StageMemoryError`` -- before any peer starts sending tensors;
4. on accelerators, optionally *probe*: allocate the estimated remainder
   once and free it, turning a later mid-step OOM into a clean startup
   failure the coordinator can replan around.

The report looks like::

    estimated = 3.82 GB, actual_before_training = 4.11 GB, error = +7.6%
"""

from __future__ import annotations

import gc

import torch

GB = 1024**3


class StageMemoryError(RuntimeError):
    """A stage cannot fit on its device. ``report`` carries the measurements."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, StageMemoryError):
        return True
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def release_device_memory(device) -> None:
    gc.collect()
    if device.backend == "cuda":
        torch.cuda.empty_cache()
    elif device.backend == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def validate_stage(stage, device, estimate: dict | None, *, probe: bool = True,
                   safety_factor: float = 0.85) -> dict:
    """Measure a freshly materialised stage; raise StageMemoryError if unsafe."""
    held = stage.memory_report()
    actual_params = held["parameters"]
    available = device.available_memory(in_use=actual_params)
    stats = device.memory_stats()
    report = {
        "backend": device.backend,
        "actual_parameters": actual_params,
        "actual_before_training": int(stats.get("allocated", actual_params)) if device.backend != "cpu"
        else actual_params,
        "available": int(available),
        "capacity": int(actual_params + available),
        "probe": None,
    }
    if estimate:
        est_required, est_params = int(estimate["required"]), int(estimate["parameters"])
        remaining = max(0, est_required - est_params)
        report.update(estimated_required=est_required, estimated_parameters=est_params,
                      estimated_remaining=remaining,
                      parameter_estimate_error=(actual_params - est_params) / est_params if est_params else 0.0)
        if device.backend != "cpu" and est_params:
            before = report["actual_before_training"]
            report["error_before_training"] = (before - est_params) / est_params
        if remaining > available * safety_factor:
            raise StageMemoryError(
                f"stage {stage.stage_index} needs ~{remaining / GB:.2f} GB more after loading its "
                f"{actual_params / GB:.2f} GB of parameters, but the {device.backend} device has only "
                f"{available / GB:.2f} GB available (safety factor {safety_factor})", report)
        if probe and device.backend in ("cuda", "mps") and remaining > 0:
            try:
                blocks = [torch.empty(min(remaining - i, 256 * 1024**2), dtype=torch.uint8, device=device.device)
                          for i in range(0, remaining, 256 * 1024**2)]
                device.synchronize()
                report["probe"] = "ok"
                del blocks
            except Exception as exc:
                if not is_oom(exc):
                    raise
                report["probe"] = "oom"
                release_device_memory(device)
                raise StageMemoryError(f"probe allocation of {remaining / GB:.2f} GB failed on "
                                       f"{device.backend}: {str(exc).splitlines()[0][:160]}", report) from exc
            release_device_memory(device)
    return report


def format_validation(report: dict) -> str:
    if "estimated_required" not in report:
        return f"actual parameters {report['actual_parameters'] / GB:.2f} GB, available {report['available'] / GB:.2f} GB"
    parts = [f"estimated = {report['estimated_required'] / GB:.2f} GB",
             f"parameters estimated {report['estimated_parameters'] / GB:.2f} GB vs actual "
             f"{report['actual_parameters'] / GB:.2f} GB ({report['parameter_estimate_error']:+.1%})"]
    if "error_before_training" in report:
        parts.append(f"actual_before_training = {report['actual_before_training'] / GB:.2f} GB "
                     f"(error vs estimated parameters {report['error_before_training']:+.1%})")
    if "actual_peak_step0" in report:
        parts.append(f"actual peak (first step) = {report['actual_peak_step0'] / GB:.2f} GB "
                     f"(error vs estimate {report['peak_estimate_error']:+.1%})")
    return ", ".join(parts)

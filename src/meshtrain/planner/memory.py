"""Memory accounting for a pipeline stage (V1.5).

V1 counted parameters, gradients, optimizer state, saved activations and a
small temporary allowance against ``total - headroom``. On real hardware
(GTX 1050 Ti, 4 GB) that accepted a split that then hit CUDA OOM. V1.5
accounts for every category separately and makes the safety margin
explicit and per-backend.

For a stage with layers L, P = parameter bytes, on a device with T bytes:

    usable                    = T * safety_factor[backend] - framework_reserve[backend]
    parameters                = P
    gradients                 = P                       (.grad kept across microbatches)
    optimizer_state           = k * P                   k = 0 sgd, 2 adam/adamw (exp_avg, exp_avg_sq)
    optimizer_step_temporary  = t * P                   t = 0 sgd, 1 adam/adamw: the multi-tensor
                                                        ("foreach") update materialises a
                                                        parameter-sized intermediate (denominator)
    master_weights            = 0                       (V1.5 trains in one dtype; no fp32 master copy)
    in_flight                 = microbatch graphs held at once:
                                  last stage            1
                                  gpipe                 M
                                  1f1b                  min(S - s, M, max_inflight)
    saved_activations         = in_flight * Σ saved_bytes(L) * activation_safety
    input_buffers             = in_flight * input_bytes(first layer)   (received boundary inputs)
    output_buffers            = in_flight * output_bytes(last layer)   (outputs kept until backward)
                                + 2 * input_bytes                      (input-gradients awaiting send)
    transport_buffers         = CPU backend: 2 * max(boundary bytes)   (pooled host buffers live in the
                                same memory); accelerators: 0 on the device (pinned host staging is
                                reported as host_staging, outside the device budget)
    temporary_workspace       = 2 * max output_bytes + max per-layer saved bytes
                                (one layer's backward re-materialises gradients of what it saved)

    fits  <=>  Σ components <= usable
               usable is further capped at the live ``available`` memory
               (free + reclaimable allocator cache) when a worker reports it

Defaults (all configurable, see config ``memory:``): safety_factor cuda 0.85,
mps 0.80, cpu 0.85; framework_reserve cuda 0.4 GB (CUDA context, cuBLAS/cuDNN
workspaces), mps 0 (unified memory, budget already reduced by the
recommended working-set size), cpu 0.25 GB. These are starting points, not
universal truths: runtime validation (runtime/memory_check.py) measures the
real numbers and records the estimation error so they can be calibrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

OPTIMIZER_STATE_FACTOR = {"sgd": 0.0, "adam": 2.0, "adamw": 2.0}
OPTIMIZER_TEMP_FACTOR = {"sgd": 0.0, "adam": 1.0, "adamw": 1.0}
GB = 1024**3
DEFAULT_SAFETY_FACTOR = {"cuda": 0.85, "mps": 0.80, "cpu": 0.85}
DEFAULT_FRAMEWORK_RESERVE = {"cuda": int(0.4 * GB), "mps": 0, "cpu": int(0.25 * GB)}


@dataclass
class MemoryEstimate:
    total: int
    usable: int
    parameters: int
    gradients: int
    optimizer_state: int
    optimizer_step_temporary: int = 0
    master_weights: int = 0
    saved_activations: int = 0
    input_buffers: int = 0
    output_buffers: int = 0
    transport_buffers: int = 0
    temporary_workspace: int = 0
    framework_reserve: int = 0
    host_staging: int = 0
    in_flight: int = 1
    available: int | None = None   # live free + reclaimable cache from a heartbeat (None = unknown)
    residency: dict | None = None  # V2: residency plan (HOT/COLD layers) behind these numbers

    COMPONENTS = ("parameters", "gradients", "optimizer_state", "optimizer_step_temporary", "master_weights",
                  "saved_activations", "input_buffers", "output_buffers", "transport_buffers",
                  "temporary_workspace")

    @property
    def required(self) -> int:
        return sum(getattr(self, k) for k in self.COMPONENTS)

    @property
    def budget(self) -> int:
        return self.usable

    @property
    def reserved(self) -> int:
        """Bytes of the device deliberately left unused (safety margin + framework)."""
        return self.total - self.usable

    @property
    def temporary(self) -> int:  # V1 name
        return self.temporary_workspace

    @property
    def fits(self) -> bool:
        return self.required <= self.usable

    def to_dict(self) -> dict:
        return {**asdict(self), "required": self.required, "budget": self.budget, "reserved": self.reserved,
                "fits": self.fits}

    def format(self, worker: str, unified: bool = False) -> str:
        gb = lambda n: f"{n / GB:8.2f} GB"  # noqa: E731
        star = "*" if unified else ""
        rows = [(f"total accelerator memory{star}", self.total)]
        if self.available is not None:
            rows.append(("available (live, incl. cache)", self.available))
        rows.append(("usable (after safety/framework)", self.usable))
        rows += [(k.replace("_", " "), getattr(self, k)) for k in self.COMPONENTS if getattr(self, k)]
        rows += [("required", self.required), ("free after plan", self.usable - self.required)]
        out = [f"Worker: {worker}"] + [f"    {k:<34}{gb(v)}" for k, v in rows]
        if self.host_staging:
            out.append(f"    {'host staging (pinned RAM)':<34}{gb(self.host_staging)}")
        if unified:
            out.append("    * unified memory shared with the OS")
        return "\n".join(out)


def reserved_bytes(total: int, headroom_fraction: float, headroom_min_bytes: int) -> int:
    return int(max(total * headroom_fraction, headroom_min_bytes))


def in_flight_microbatches(schedule: str, stage_index: int, num_stages: int, num_microbatches: int,
                           max_inflight: int | None = None) -> int:
    if stage_index >= num_stages - 1:
        return 1
    if schedule == "1f1b":
        cap = num_microbatches if max_inflight is None else max_inflight
        return max(1, min(num_stages - stage_index, num_microbatches, cap))
    return num_microbatches


def estimate_stage_memory(layers, *, total: int, optimizer: str, num_microbatches: int, is_last: bool,
                          stage_index: int | None = None, num_stages: int | None = None,
                          schedule: str = "gpipe", max_inflight: int | None = None, backend: str = "cuda",
                          safety_factor: float | None = None, framework_reserve: int | None = None,
                          headroom_fraction: float | None = None, headroom_min_bytes: int | None = None,
                          activation_safety: float = 1.25, available: int | None = None) -> MemoryEstimate:
    """Estimate one stage's device memory (see module docstring).

    ``headroom_fraction``/``headroom_min_bytes`` reproduce V1's margin
    (``reserved = max(fraction*T, min)``) when given; otherwise the per-backend
    ``safety_factor``/``framework_reserve`` apply. ``available`` (live free
    memory) caps the usable budget when known.
    """
    if num_stages is None:  # V1 call style: only "is_last" known
        num_stages, stage_index = (1, 0) if is_last else (2, 0)
    stage_index = stage_index if stage_index is not None else (num_stages - 1 if is_last else 0)
    P = sum(l.param_bytes for l in layers)
    in_flight = in_flight_microbatches(schedule, stage_index, num_stages, num_microbatches, max_inflight)
    if is_last:
        in_flight = 1
    saved = sum(l.saved_bytes for l in layers)
    in_bytes, out_bytes = layers[0].input_bytes, layers[-1].activation_bytes
    boundary = max(in_bytes, out_bytes)
    if headroom_fraction is not None or headroom_min_bytes is not None:
        reserve = reserved_bytes(total, headroom_fraction or 0.0, headroom_min_bytes or 0)
        usable = int(total) - reserve
        fw = 0
    else:
        sf = DEFAULT_SAFETY_FACTOR.get(backend, 0.85) if safety_factor is None else safety_factor
        fw = DEFAULT_FRAMEWORK_RESERVE.get(backend, 0) if framework_reserve is None else framework_reserve
        usable = int(total * sf) - fw
    if available is not None:
        usable = max(0, min(usable, int(available)))
    return MemoryEstimate(
        total=int(total),
        usable=usable,
        parameters=P,
        gradients=P,
        optimizer_state=int(P * OPTIMIZER_STATE_FACTOR[optimizer]),
        optimizer_step_temporary=int(P * OPTIMIZER_TEMP_FACTOR[optimizer]),
        master_weights=0,
        saved_activations=int(in_flight * saved * activation_safety),
        input_buffers=in_flight * in_bytes,
        output_buffers=(0 if is_last else in_flight * out_bytes) + 2 * in_bytes,
        transport_buffers=2 * boundary if backend == "cpu" else 0,
        temporary_workspace=2 * max(l.activation_bytes for l in layers) + max(l.saved_bytes for l in layers),
        framework_reserve=fw,
        host_staging=0 if backend == "cpu" else (in_flight + 2) * boundary,
        in_flight=in_flight,
        available=available,
    )

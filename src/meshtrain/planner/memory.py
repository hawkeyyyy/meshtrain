"""Memory accounting for a pipeline stage (approximate, documented).

For a stage holding layers L on a worker with total memory T:

    reserved          = max(headroom_fraction * T, headroom_min)   # runtime/context/fragmentation
    parameters        = sum param_bytes
    gradients         = parameters                                  # .grad persists across microbatches
    optimizer_state   = k * parameters,  k = 0 (sgd), 2 (adam/adamw: exp_avg + exp_avg_sq)
    saved_activations = in_flight * (input_bytes + sum saved_bytes) * safety
        in_flight = M for every stage except the last (GPipe: a stage holds
                    all M microbatch graphs until gradients return), 1 for the last
                    stage (it runs backward immediately).
    temporary         = 2 * max(activation_bytes) of one microbatch (send staging + workspace)

    fits  <=>  parameters + gradients + optimizer_state + saved_activations + temporary
               <= T - reserved

Approximations: caching-allocator fragmentation, CUDA context size and
kernel workspaces are covered only by ``reserved``; unified-memory (MPS)
devices share T with the OS and other processes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

OPTIMIZER_STATE_FACTOR = {"sgd": 0.0, "adam": 2.0, "adamw": 2.0}


@dataclass
class MemoryEstimate:
    total: int
    reserved: int
    parameters: int
    gradients: int
    optimizer_state: int
    saved_activations: int
    temporary: int

    @property
    def required(self) -> int:
        return self.parameters + self.gradients + self.optimizer_state + self.saved_activations + self.temporary

    @property
    def budget(self) -> int:
        return self.total - self.reserved

    @property
    def fits(self) -> bool:
        return self.required <= self.budget

    def to_dict(self) -> dict:
        return {**asdict(self), "required": self.required, "budget": self.budget, "fits": self.fits}

    def format(self, worker: str, unified: bool = False) -> str:
        gb = lambda n: f"{n / 1024**3:8.2f} GB"  # noqa: E731
        star = "*" if unified else ""
        rows = [
            (f"total accelerator memory{star}", self.total),
            ("reserved (headroom)", self.reserved),
            ("parameters", self.parameters),
            ("gradients", self.gradients),
            ("optimizer state", self.optimizer_state),
            ("saved activations (est.)", self.saved_activations),
            ("temporary (est.)", self.temporary),
            ("required", self.required),
            ("free after plan", self.budget - self.required),
        ]
        out = [f"Worker: {worker}"] + [f"    {k:<30}{gb(v)}" for k, v in rows]
        if unified:
            out.append("    * unified memory shared with the OS")
        return "\n".join(out)


def reserved_bytes(total: int, headroom_fraction: float, headroom_min_bytes: int) -> int:
    return int(max(total * headroom_fraction, headroom_min_bytes))


def estimate_stage_memory(layers, *, total: int, optimizer: str, num_microbatches: int, is_last: bool,
                          headroom_fraction: float = 0.15, headroom_min_bytes: int = 512 * 1024**2,
                          activation_safety: float = 1.25) -> MemoryEstimate:
    params = sum(l.param_bytes for l in layers)
    in_flight = 1 if is_last else num_microbatches
    per_mb = layers[0].input_bytes + sum(l.saved_bytes for l in layers)
    return MemoryEstimate(
        total=int(total),
        reserved=reserved_bytes(total, headroom_fraction, headroom_min_bytes),
        parameters=params,
        gradients=params,
        optimizer_state=int(params * OPTIMIZER_STATE_FACTOR[optimizer]),
        saved_activations=int(in_flight * per_mb * activation_safety),
        temporary=2 * max(l.activation_bytes for l in layers),
    )

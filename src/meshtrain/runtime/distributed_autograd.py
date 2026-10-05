"""Explicit distributed-autograd boundaries.

PyTorch's autograd graph never crosses a machine boundary. Instead:

    sender   (stage k)   : h = f_k(x);  save (x, h);  send h.detach()
    receiver (stage k+1) : x' = recv().to(dev).detach().requires_grad_(True)
                           ...local forward/backward...
                           send x'.grad upstream
    sender   (stage k)   : h.backward(received_grad)   # fills dL/dθ_k and x.grad

``MicrobatchContext`` keeps the local graph of one microbatch alive between
its forward and backward. ``ContextStore`` indexes contexts by
(step_id, microbatch_id) so a returning gradient always meets the activation
it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


class BoundaryError(RuntimeError):
    """Gradient/activation routing violated the pipeline protocol."""


def make_boundary_input(received: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    """Turn a received activation into a fresh autograd leaf on this stage.

    Integer inputs (e.g. token ids) can never require grad.
    """
    leaf = received.detach()
    if requires_grad and leaf.is_floating_point():
        leaf.requires_grad_(True)
    return leaf


def boundary_backward(output: torch.Tensor, grad_output: torch.Tensor) -> None:
    """Continue backprop from a remote gradient into this stage's local graph."""
    if grad_output.shape != output.shape:
        raise BoundaryError(f"gradient shape {tuple(grad_output.shape)} != activation shape {tuple(output.shape)}")
    if grad_output.device.type != output.device.type:
        raise BoundaryError(f"gradient on {grad_output.device}, activation on {output.device}")
    output.backward(grad_output.to(output.dtype))


def boundary_input_grad(inp: torch.Tensor) -> torch.Tensor | None:
    """Gradient to send upstream (None if the input did not need grad)."""
    if not inp.requires_grad:
        return None
    if inp.grad is None:
        # Input not used by the stage: the true gradient is zero.
        return torch.zeros_like(inp)
    return inp.grad


@dataclass
class MicrobatchContext:
    step_id: int
    microbatch_id: int
    input: torch.Tensor | None = None
    output: torch.Tensor | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int]:
        return (self.step_id, self.microbatch_id)

    def saved_bytes(self) -> int:
        n = 0
        for t in (self.input, self.output):
            if t is not None:
                n += t.numel() * t.element_size()
        return n


class ContextStore:
    def __init__(self) -> None:
        self._contexts: dict[tuple[int, int], MicrobatchContext] = {}

    def put(self, ctx: MicrobatchContext) -> None:
        if ctx.key in self._contexts:
            raise BoundaryError(f"duplicate forward for step={ctx.step_id} microbatch={ctx.microbatch_id}")
        self._contexts[ctx.key] = ctx

    def pop(self, step_id: int, microbatch_id: int) -> MicrobatchContext:
        try:
            return self._contexts.pop((step_id, microbatch_id))
        except KeyError:
            raise BoundaryError(
                f"gradient for unknown step={step_id} microbatch={microbatch_id}; "
                f"pending={sorted(self._contexts)}"
            ) from None

    def __len__(self) -> int:
        return len(self._contexts)

    def pending(self) -> list[tuple[int, int]]:
        return sorted(self._contexts)

    def saved_bytes(self) -> int:
        return sum(c.saved_bytes() for c in self._contexts.values())

    def clear(self) -> None:
        self._contexts.clear()

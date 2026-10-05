"""A pipeline stage: a contiguous slice of the model on one device.

The stage owns its module, optimizer, device adapter and saved microbatch
contexts. It knows nothing about networking: callers pass tensors in and get
tensors out.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
from torch import nn

from meshtrain.runtime.distributed_autograd import (
    BoundaryError,
    ContextStore,
    MicrobatchContext,
    boundary_backward,
    boundary_input_grad,
    make_boundary_input,
)
from meshtrain.worker.device import CPUDeviceAdapter, DeviceAdapter


def build_optimizer(name: str, params, lr: float, **kwargs) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=kwargs.get("momentum", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=kwargs.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {name!r}")


class Stage:
    def __init__(
        self,
        module: nn.Module,
        *,
        stage_index: int,
        num_stages: int,
        device: DeviceAdapter | None = None,
        optimizer: str = "sgd",
        lr: float = 0.01,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        name: str | None = None,
    ):
        self.device = device or CPUDeviceAdapter()
        self.module = self.device.move_module(module)
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.name = name or f"stage{stage_index}"
        self.optimizer = build_optimizer(optimizer, self.module.parameters(), lr) if any(
            True for _ in self.module.parameters()) else None
        self.loss_fn = loss_fn
        self.contexts = ContextStore()
        self._param_ids = {id(p) for p in self.module.parameters()}
        if self.is_last and loss_fn is None:
            raise ValueError("the last stage needs a loss function")

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.num_stages - 1

    # ------------------------------------------------------------------
    def forward(self, inp: torch.Tensor, context: MicrobatchContext) -> torch.Tensor:
        """Run the local forward; saves the graph in ``context``.

        Returns the *attached* output; callers send ``output.detach()``.
        """
        t0 = time.perf_counter()
        x = make_boundary_input(self.device.move_tensor(inp), requires_grad=not self.is_first)
        param_ids = self._param_ids
        saved = 0

        def pack(t):
            nonlocal saved
            base = t._base if t._base is not None else t
            if id(t) not in param_ids and id(base) not in param_ids:
                saved += t.numel() * t.element_size()
            return t

        # Count what autograd keeps alive for backward (excluding parameters).
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = self.module(x)
        context.autograd_saved_bytes = saved
        self.device.sync_compute()
        context.input, context.output = x, out
        context.timings["forward"] = time.perf_counter() - t0
        self.contexts.put(context)
        return out

    def forward_loss(self, inp: torch.Tensor, target: torch.Tensor, context: MicrobatchContext,
                     loss_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Last stage: forward, loss and local backward in one go.

        Returns (unscaled loss value, gradient w.r.t. the stage input).
        """
        if not self.is_last:
            raise BoundaryError("forward_loss is only valid on the last stage")
        out = self.forward(inp, context)
        context.peak_saved_bytes = self.contexts.saved_bytes()
        t0 = time.perf_counter()
        loss = self.loss_fn(out, self.device.move_tensor(target))
        self.contexts.pop(context.step_id, context.microbatch_id)
        (loss * loss_scale).backward()
        self.device.sync_compute()
        context.timings["backward"] = time.perf_counter() - t0
        grad = boundary_input_grad(context.input)
        context.input = context.output = None
        return loss.detach(), grad

    def loss_backward(self, target: torch.Tensor, context_key: tuple[int, int],
                      loss_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor | None, MicrobatchContext]:
        """Last stage: loss on the saved output of one microbatch, then backward.

        Returns (unscaled loss, gradient w.r.t. the stage input, context).
        """
        if not self.is_last:
            raise BoundaryError("loss_backward is only valid on the last stage")
        ctx = self.contexts.pop(*context_key)
        t0 = time.perf_counter()
        loss = self.loss_fn(ctx.output, self.device.move_tensor(target))
        (loss * loss_scale).backward()
        self.device.sync_compute()
        ctx.timings["backward"] = time.perf_counter() - t0
        grad = boundary_input_grad(ctx.input) if not self.is_first else None
        ctx.input = ctx.output = None
        return loss.detach(), grad, ctx

    def backward(self, grad_output: torch.Tensor, context_key: tuple[int, int]) -> tuple[torch.Tensor | None, MicrobatchContext]:
        """Backprop a received gradient through the saved graph of one microbatch."""
        ctx = self.contexts.pop(*context_key)
        t0 = time.perf_counter()
        boundary_backward(ctx.output, self.device.move_tensor(grad_output))
        self.device.sync_compute()
        ctx.timings["backward"] = time.perf_counter() - t0
        grad = boundary_input_grad(ctx.input) if not self.is_first else None
        ctx.input = ctx.output = None
        return grad, ctx

    def optimizer_step(self) -> float:
        if len(self.contexts):
            raise BoundaryError(f"optimizer step with pending microbatches {self.contexts.pending()}")
        t0 = time.perf_counter()
        if self.optimizer is not None:
            self.optimizer.step()
        self.device.sync_compute()
        return time.perf_counter() - t0

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.module.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    def named_gradients(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}{n}": p.grad.detach().cpu().clone()
                for n, p in self.module.named_parameters() if p.grad is not None}

    def named_parameters_cpu(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}{n}": p.detach().cpu().clone() for n, p in self.module.named_parameters()}

    def memory_report(self) -> dict[str, int]:
        params = sum(p.numel() * p.element_size() for p in self.module.parameters())
        grads = sum(p.grad.numel() * p.grad.element_size() for p in self.module.parameters() if p.grad is not None)
        opt = 0
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        opt += v.numel() * v.element_size()
        return {"parameters": params, "gradients": grads, "optimizer_state": opt,
                "saved_activations": self.contexts.saved_bytes()}

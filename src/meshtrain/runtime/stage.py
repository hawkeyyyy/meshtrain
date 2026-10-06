"""A pipeline stage: a contiguous slice of the model on one device.

The stage owns its module, optimizer, device adapter and saved microbatch
contexts. It knows nothing about networking: callers pass tensors in and get
tensors out.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

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
from meshtrain.runtime.offload import _ParamRef
from meshtrain.runtime.tensor_store import LocalTensorStore, TensorRole, activation_id, parameter_id
from meshtrain.worker.device import CPUDeviceAdapter, DeviceAdapter

if TYPE_CHECKING:
    from meshtrain.runtime.offload import ResidencyPolicy


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
        layer_offset: int = 0,
        residency: "ResidencyPolicy | None" = None,
        accelerator_budget: int | None = None,
    ):
        self.device = device or CPUDeviceAdapter()
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.name = name or f"stage{stage_index}"
        self.layer_offset = layer_offset
        self.residency_policy = residency
        self.accelerator_budget = accelerator_budget
        self.residency = None  # ResidencyManager when a V2 offload policy is active
        if residency is not None and accelerator_budget is not None and residency.enforce_allocator_limit:
            self.device.limit_memory(accelerator_budget)
        offload = residency is not None and residency.active
        self.tensor_store = LocalTensorStore(self.name, self.device.backend, device=self.device,
                                             compute_stage=stage_index, accelerator_budget=accelerator_budget,
                                             offload=offload)
        if offload:
            from meshtrain.runtime.offload import ResidencyManager

            for p in module.parameters():
                if not self.device.supports_dtype(p.dtype):
                    raise TypeError(f"{self.device.backend} does not support parameter dtype {p.dtype}")
            self.module = module
            self.residency = ResidencyManager(module, self.device, self.tensor_store, residency,
                                              layer_offset=layer_offset, optimizer=optimizer)
        else:
            self.module = self.device.move_module(module)
        self.optimizer = build_optimizer(optimizer, self.module.parameters(), lr) if any(
            True for _ in self.module.parameters()) else None
        self.loss_fn = loss_fn
        self.contexts = ContextStore()
        self._param_ids = {id(p) for p in self.module.parameters()}
        if self.is_last and loss_fn is None:
            raise ValueError("the last stage needs a loss function")
        # Stable identities for everything this stage owns.
        if not offload:
            for n, p in self.module.named_parameters():
                self.tensor_store.put(self.global_name(n), p, TensorRole.PARAMETER)

    def global_name(self, local_name: str) -> str:
        """'3.qkv.weight' inside this stage -> 'model.layers.<offset+3>.qkv.weight'."""
        idx, rest = local_name.split(".", 1)
        return parameter_id(int(idx) + self.layer_offset, rest)

    def refresh_tensor_store(self) -> None:
        """Record gradients and optimizer state (they appear lazily during training)."""
        if self.residency is not None:
            self.residency.refresh_records(self.optimizer)
            return
        store = self.tensor_store
        for n, p in self.module.named_parameters():
            gid = self.global_name(n)
            if p.grad is not None:
                store.put(gid + ".grad", p.grad, TensorRole.GRADIENT)
            if self.optimizer is not None:
                for key, v in self.optimizer.state.get(p, {}).items():
                    if torch.is_tensor(v) and v.dim() > 0:
                        store.put(f"{gid}.optim.{key}", v, TensorRole.OPTIMIZER_STATE)

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

        unpack = _identity
        res = self.residency
        if res is not None:
            # Offloaded parameters are saved by reference and reloaded on demand in backward.
            count = pack

            def pack(t):  # noqa: F811
                ref = res.pack(t)
                return count(t) if ref is None else ref

            def unpack(obj):
                return res.unpack(obj) if isinstance(obj, _ParamRef) else obj

        # Count what autograd keeps alive for backward (excluding parameters).
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = self.module(x)
        context.autograd_saved_bytes = saved
        self.device.sync_compute()
        context.input, context.output = x, out
        context.timings["forward"] = time.perf_counter() - t0
        self.contexts.put(context)
        self.tensor_store.put(activation_id(context.step_id, context.microbatch_id, self.stage_index), x,
                              TensorRole.ACTIVATION)
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
        self.tensor_store.discard(activation_id(context.step_id, context.microbatch_id, self.stage_index))
        (loss * loss_scale).backward()
        self._after_backward()
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
        self.tensor_store.discard(activation_id(*context_key, self.stage_index))
        t0 = time.perf_counter()
        loss = self.loss_fn(ctx.output, self.device.move_tensor(target))
        (loss * loss_scale).backward()
        self._after_backward()
        self.device.sync_compute()
        ctx.timings["backward"] = time.perf_counter() - t0
        grad = boundary_input_grad(ctx.input) if not self.is_first else None
        ctx.input = ctx.output = None
        return loss.detach(), grad, ctx

    def backward(self, grad_output: torch.Tensor, context_key: tuple[int, int]) -> tuple[torch.Tensor | None, MicrobatchContext]:
        """Backprop a received gradient through the saved graph of one microbatch."""
        ctx = self.contexts.pop(*context_key)
        self.tensor_store.discard(activation_id(*context_key, self.stage_index))
        t0 = time.perf_counter()
        boundary_backward(ctx.output, self.device.move_tensor(grad_output))
        self._after_backward()
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
            if self.residency is not None:
                self.residency.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
        self.device.sync_compute()
        return time.perf_counter() - t0

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.module.zero_grad(set_to_none=True)
        if self.residency is not None:
            self.residency.zero_grad()

    def _after_backward(self) -> None:
        if self.residency is not None:
            self.residency.after_backward_call()

    def close(self) -> None:
        """Release hooks and the allocator cap (worker processes run several jobs)."""
        if self.residency is not None:
            self.residency.close()
        if self.residency_policy is not None and self.accelerator_budget is not None:
            self.device.limit_memory(None)

    # ------------------------------------------------------------------
    def _grad(self, p) -> torch.Tensor | None:
        return self.residency.gradient(p) if self.residency is not None else p.grad

    def named_gradients(self, prefix: str = "") -> dict[str, torch.Tensor]:
        out = {}
        for n, p in self.module.named_parameters():
            g = self._grad(p)
            if g is not None:
                out[f"{prefix}{n}"] = g.detach().cpu().clone()
        return out

    def named_parameters_cpu(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}{n}": p.detach().cpu().clone() for n, p in self.module.named_parameters()}

    def memory_report(self) -> dict[str, int]:
        params = sum(p.numel() * p.element_size() for p in self.module.parameters())
        grads = sum(g.numel() * g.element_size() for g in map(self._grad, self.module.parameters()) if g is not None)
        opt = 0
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        opt += v.numel() * v.element_size()
        return {"parameters": params, "gradients": grads, "optimizer_state": opt,
                "saved_activations": self.contexts.saved_bytes()}


def _identity(t):
    return t

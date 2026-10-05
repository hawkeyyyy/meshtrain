"""Layer-level profile of a sequential model, computed on the ``meta`` device.

No real memory is allocated: every layer is instantiated on ``meta`` and a
meta input of one microbatch is pushed through, which gives exact shapes.

Per layer we record:

* ``param_bytes``          -- parameter storage
* ``activation_bytes``     -- output tensor size (what crosses a stage boundary)
* ``saved_bytes``          -- bytes autograd saves for backward inside the layer,
                              measured with ``saved_tensors_hooks``
* ``flops``                -- forward FLOPs (torch FlopCounterMode), per microbatch
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from meshtrain.models.base import ModelSpec


@dataclass
class LayerProfile:
    index: int
    name: str
    param_count: int
    param_bytes: int
    input_bytes: int
    activation_bytes: int
    saved_bytes: int
    flops: float
    output_shape: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def profile_model(spec: ModelSpec, microbatch_size: int) -> list[LayerProfile]:
    x, _ = spec.make_batch(0, microbatch_size)
    x = x.to("meta")
    profiles = []
    for i in range(spec.num_layers):
        with torch.device("meta"):
            layer = spec._make_layer(i).to(spec.dtype)
        inp = x.detach()
        if inp.is_floating_point():
            inp.requires_grad_(True)
        saved: dict[int, int] = {}
        param_ids = {id(p) for p in layer.parameters()}

        def pack(t):
            base = t._base if t._base is not None else t
            if id(t) not in param_ids and id(base) not in param_ids:  # params counted separately
                saved[id(t)] = _nbytes(t)
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            with FlopCounterMode(display=False) as counter:
                out = layer(inp)
        params = list(layer.parameters())
        param_bytes = sum(_nbytes(p) for p in params)
        saved_bytes = sum(saved.values())
        profiles.append(LayerProfile(
            index=i,
            name=spec.layer_name(i),
            param_count=sum(p.numel() for p in params),
            param_bytes=param_bytes,
            input_bytes=_nbytes(x),
            activation_bytes=_nbytes(out),
            saved_bytes=saved_bytes,
            flops=float(counter.get_total_flops()),
            output_shape=tuple(out.shape),
        ))
        x = out.detach()
    return profiles

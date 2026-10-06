"""Residency-independent stage checkpoints (V2).

A stage may hold some parameters on the device, some in host RAM and (later)
some remotely. ``save_stage_checkpoint`` gathers the *logical* state through
each tensor's authoritative copy -- never only what happens to be on the
accelerator -- and writes it keyed by stable tensor ids::

    {"format": "meshtrain-stage-checkpoint-v1", "step": n,
     "parameters": {tensor_id: cpu tensor},
     "optimizer":  {tensor_id: {state key: cpu tensor}},
     "optimizer_hparams": [...param group hyper-parameters...],
     "residency": TensorStore manifest at save time (informational)}

``load_stage_checkpoint`` writes values into the stage's authoritative copies
for the *current* residency policy (which may differ from the saving run) and
re-creates optimizer state on the tier that policy uses (device for the
accelerator optimizer, RAM for ``cpu_offload``).
"""

from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "meshtrain-stage-checkpoint-v1"


def _named(stage):
    for n, p in stage.module.named_parameters():
        yield stage.global_name(n), p


def save_stage_checkpoint(stage, path: str | Path, step: int | None = None) -> Path:
    res = stage.residency
    params, optim = {}, {}
    remote_values = {}
    if res is not None:   # remote layers: fetch the committed params + AdamW state (V2.5)
        for g in res.remote_groups:
            remote_values.update(res.fetch_logical(g))
    for tid, p in _named(stage):
        if tid in remote_values:
            params[tid] = remote_values[tid]
            st = {k: remote_values[f"{tid}.optim.{k}"] for k in ("exp_avg", "exp_avg_sq")
                  if f"{tid}.optim.{k}" in remote_values}
            step = (stage.optimizer.state.get(p, {}) if stage.optimizer is not None else {}).get("step")
            if st:
                optim[tid] = {**st, **({"step": step.detach().clone()} if torch.is_tensor(step) else {})}
            continue
        if res is not None:
            g = res.by_param[id(p)][0]
            meta = stage.tensor_store.locate(tid)
            # Authoritative copy: RAM master unless the device copy is dirty / the layer is HOT on the device.
            src = g.host[id(p)] if (id(p) in g.host and not meta.dirty and meta.tier.value == "local_ram") else p.data
        else:
            src = p.data
        params[tid] = src.detach().to("cpu", copy=True)
        if stage.optimizer is not None and p in stage.optimizer.state:
            optim[tid] = {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                          for k, v in stage.optimizer.state[p].items()}
    hparams = [{k: v for k, v in grp.items() if k != "params"} for grp in stage.optimizer.param_groups] \
        if stage.optimizer is not None else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "step": step, "stage_index": stage.stage_index, "parameters": params,
                "optimizer": optim, "optimizer_hparams": hparams,
                "residency": stage.tensor_store.manifest()}, path)
    return path


def load_stage_checkpoint(stage, path: str | Path) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: not a MeshTrain stage checkpoint")
    res = stage.residency
    named = dict(_named(stage))
    missing = set(named) - set(ckpt["parameters"])
    if missing:
        raise KeyError(f"checkpoint lacks {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    cpu_opt = res is not None and res.cpu_opt
    with torch.no_grad():
        if res is not None:   # remote layers: write the checkpoint values as new committed versions
            for g in res.remote_groups:
                vals = {}
                for tid, p in g.params:
                    vals[tid] = ckpt["parameters"][tid]
                    st = ckpt["optimizer"].get(tid) or {}
                    for k in ("exp_avg", "exp_avg_sq"):
                        if k in st:
                            vals[f"{tid}.optim.{k}"] = st[k]
                    if "step" in st and stage.optimizer is not None:
                        stage.optimizer.state[p] = {"step": st["step"]}
                res.store_logical(g, vals)
        remote_ids = {tid for g in (res.remote_groups if res is not None else []) for tid, _ in g.params}
        for tid, p in named.items():
            if tid in remote_ids:
                continue
            value = ckpt["parameters"][tid]
            if value.shape != p.shape or value.dtype != p.dtype:
                raise ValueError(f"{tid}: checkpoint {tuple(value.shape)} {value.dtype} vs model "
                                 f"{tuple(p.shape)} {p.dtype}")
            if res is not None:
                g = res.by_param[id(p)][0]
                if id(p) in g.host:
                    g.host[id(p)].copy_(value)          # authoritative RAM master
                if not g.offloaded or p.data.device.type != "cpu" or (
                        id(p) in g.host and p.data_ptr() != g.host[id(p)].data_ptr()):
                    p.data.copy_(value.to(p.data.device))  # device copy (HOT, or a loaded COLD layer)
                stage.tensor_store.locate(tid).version += 1
            else:
                p.data.copy_(value.to(p.data.device))
            state = ckpt["optimizer"].get(tid)
            if state is not None and stage.optimizer is not None:
                if res is not None:
                    dev = torch.device("cpu") if cpu_opt else stage.device.device
                else:
                    dev = p.device
                stage.optimizer.state[p] = {
                    k: (v.to(dev) if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in state.items()}
    if stage.optimizer is not None:
        for grp, hp in zip(stage.optimizer.param_groups, ckpt.get("optimizer_hparams", [])):
            grp.update(hp)
    if res is not None:
        res.refresh_records(stage.optimizer)
    return {"step": ckpt.get("step"), "tensors": len(named)}

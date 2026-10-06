# MeshTrain V2: tensor residency (local-memory phase)

V1.5 decides **where a layer computes**. V2 also decides **where its state
lives**:

    compute_owner(layer) = the stage's worker      (unchanged, static for the job)
    residency(tensor)    = a memory tier on that worker

A layer can compute on the RTX 4070 while its weights, gradients and AdamW
state live in host RAM. Only what the next computation needs is moved onto
the GPU. The goal of this phase is **capacity, not speed**: train models whose
state does not fit in accelerator memory, correctly, at a measured cost.

Pipeline-stage ownership does not change in V2. Layers never migrate between
workers. Remote memory tiers are interfaces only (`NotImplementedError`).

```
                    Global Model
                         │
                  Pipeline Planner            (V1.5: which worker computes which layers)
                         │
           ┌─────────────┴─────────────┐
      Compute Stage A             Compute Stage B
       RTX 4070                     M2 / CPU
           │                           │
      TensorStore + ResidencyManager   TensorStore + ResidencyManager
      ┌──────────┐                     ┌──────────┐
      │ GPU  HOT │ ◀─prefetch/evict─▶  │ accel    │
      │ RAM COLD │                     │ RAM      │
      └──────────┘                     └──────────┘
              future: RemoteTensorStore (remote RAM / remote GPU)
```

## Configuration

```yaml
memory:
  strategy: auto_offload        # static (V1.5, default) | manual_offload | auto_offload
  accelerator_budget: 6GB       # or accelerator_budget_mb: 6144, or "auto" (device usable)
  enforce_allocator_limit: true # CUDA: hard-cap the caching allocator at the budget
  keep_resident: [layers.0, layers.11]   # manual_offload: layers that stay on the device
  prefetch_distance: 1          # 0 = synchronous loads
  eviction: after_use           # after_use | after_backward
  optimizer_offload: true       # alias for optimizer.execution: cpu_offload
  use_local_ram: true
  use_remote_ram: false         # V2.5, not implemented (rejected)
optimizer:
  execution: accelerator        # accelerator (V1.5) | cpu_offload (V2.3)
```

`memory.strategy: static` with `optimizer.execution: accelerator` is exactly
the V1.5 code path. `Stage` then builds no residency manager at all.

## Tensor identity and the TensorStore

Every tracked tensor has a stable global id:

    model.layers.14.qkv.weight                  parameter
    model.layers.14.qkv.weight.grad             gradient
    model.layers.14.qkv.weight.optim.exp_avg    optimizer state
    act.s3.mb2.stage1.input                     activation

`TensorMeta` records role, shape, dtype, bytes, owner worker, compute stage,
group (layer), **authoritative tier**, **cached tiers**, **dirty**, **version**,
pinned and last access. Invariant: one authoritative copy at any time.
Copies may exist, and `dirty` says whether the accelerator copy is newer than
RAM. A dirty copy can never be dropped without a writeback
(`set_residency` raises).

Tiers: `LOCAL_ACCELERATOR` and `LOCAL_RAM` are implemented.
`REMOTE_ACCELERATOR`, `REMOTE_RAM`, `LOCAL_NVME`, `REMOTE_NVME` and `RECOMPUTE`
exist in the enum, and every operation that would use them raises
`NotImplementedError`.

On a CPU-backend stage the compute memory *is* RAM. Static CPU stages report
`LOCAL_RAM` as in V1.5. An offloading CPU stage treats its working set as
`LOCAL_ACCELERATOR` logically, with real copies, so all offload logic is
testable without a GPU.

## Parameter identity across moves (the design choice)

Optimizers, autograd and modules all hold references to `nn.Parameter`
objects. V2 **never replaces a Parameter**. Each offloaded parameter has a
host master tensor (pinned on CUDA). Only `param.data` is re-pointed:

| Situation | `p.data` | Authoritative |
|---|---|---|
| COLD layer between uses | host master | RAM |
| COLD layer loaded for compute | device copy (clean) | RAM (device cached) |
| COLD layer after a device optimizer step | device copy (dirty) | device → written back before eviction |
| HOT layer, accelerator optimizer | device tensor | device (no RAM copy) |
| HOT layer, cpu_offload optimizer | device copy (clean) | RAM master |

Values are copied, never aliased across tiers. The optimizer keys its state by
the same Parameter objects, so it sees one parameter per tensor whatever the
residency. `Module.to()` itself uses `param.data` assignment, so this is a
supported PyTorch path.

## Lifecycle of an offloaded layer

```
RESIDENT_RAM ─▶ PREFETCHING ─▶ RESIDENT_ACCELERATOR ─▶ IN_USE_FORWARD ─▶ SAVED_FOR_BACKWARD
     ▲                                  │   ▲                                   │
     └──────── evict (clean) ───────────┘   └── IN_USE_BACKWARD ◀───────────────┘
                                       │
                          optimizer step on device
                                       ▼
                         DIRTY_ACCELERATOR ─▶ WRITEBACK ─▶ RESIDENT_ACCELERATOR
```

Transitions are explicit and validated (`ParameterGroup.transition`). An
illegal edge raises. Layers are never evicted while `IN_USE_*`, and HOT
layers are never evicted. Python reference counts are not used to decide
lifetimes.

## Forward, backward and gradients

* **Forward pre-hook**: make the layer device-resident (wait for its prefetch, or
  load synchronously), then prefetch the next `prefetch_distance` offloaded
  layers.
* **Saved tensors**: during a stage forward, a `saved_tensors_hooks` pack
  replaces every saved COLD parameter (or a view of one, e.g. `weight.t()`)
  with a reference. Unpacking it in backward reloads the layer on demand. The
  autograd graph therefore never pins a device copy, so `eviction: after_use`
  can drop the layer right after its forward and backward stays correct.
* **Gradients**: a post-accumulate-grad hook sees when every parameter of a COLD
  layer has its gradient for the current microbatch. It then adds them to
  host-RAM accumulators (same order of fp32 additions as on the device, so
  results are bit-identical), frees the device gradients and evicts per policy.
* `eviction: after_backward` keeps a loaded layer until its last pending
  backward. This is the conservative mode: fewer transfers, more memory.

## Optimizer

* `accelerator` (V1.5): HOT parameters step on the device. Each COLD layer is
  loaded, its accumulated gradients copied up, stepped on the device (DIRTY),
  written back to RAM and evicted. COLD optimizer state stays on the device.
* `cpu_offload` (V2.3): masters, gradients and AdamW state of *every* layer live
  in RAM. The update runs on the CPU, and HOT device copies are refreshed from
  the masters afterwards. Device memory holds only weights in use, gradients
  of HOT layers and activations.

On CUDA, a CPU AdamW step is not bit-identical to CUDA AdamW. The CPU and
CUDA kernels round differently, and Adam magnifies that on parameters whose
true gradient is about 0 (√v ≪ eps, e.g. attention key biases, which softmax
ignores). [v2-results.md](v2-results.md) gives the measured size.

## Memory budget

`accelerator_budget` is the total device memory a stage process may use.
Two layers enforce it:

1. **TensorStore ledger** (deterministic, all backends). At startup the manager
   reserves everything that exists for the whole job: HOT parameters,
   gradients and optimizer state, device-side optimizer state of COLD layers,
   one layer of gradient workspace and the optimizer-step workspace. Each load
   reserves its bytes while in flight and while resident. Old + new copies
   during a transfer are both counted. A load that does not fit evicts idle
   layers (LRU), and otherwise fails with `MemoryBudgetError`: tensor id,
   source and destination tier, requested and available bytes. Activations are
   tracked but not ledger-enforced. The planner reserves an allowance for them.
2. **CUDA allocator cap** (`enforce_allocator_limit`, default on):
   `set_per_process_memory_fraction(budget / total)`. Everything, including
   activations and fragmentation, must fit, or PyTorch raises CUDA OOM. This
   makes artificial budgets reproducible (a "2 GB GPU" on an 8 GB card) and
   stops the Windows driver from spilling into shared system memory.

## Automatic residency (V2.4, `planner/residency.py`)

Device memory model for a stage:

    HOT layer   : P + P (gradients) + k·P (optimizer state, accelerator optimizer)
    COLD layer  : k·P on the device (accelerator optimizer) or nothing (cpu_offload)
    working set : (1 + prefetch_distance) · largest COLD layer + one layer of gradient
                  workspace + optimizer-step workspace
    activations : the V1.5 estimate
    fits ⇔ sum ≤ budget · 0.90       (10%: the CUDA cap applies to *reserved* memory)

Heuristic (not optimal): start with every layer COLD. If that does not fit,
the model cannot train on this budget. Otherwise make layers HOT in decreasing
order of

    benefit = transfer time saved per step if HOT / extra device bytes if HOT

while everything fits. Transfer time saved = loads per step · P / H2D bandwidth
(+ writeback). With `after_use` a COLD layer is loaded 2·M times per step
(M microbatches), plus once for the accelerator optimizer step. Exposed load
time is modeled as `max(0, load − compute of the previous layer)` when
prefetching. Bandwidths come from the worker's measured `h2d_Bps`/`d2h_Bps`
when available.

In cluster jobs the coordinator's partition planner uses this model for each
candidate stage. A model that is infeasible under `static` can therefore become
feasible under `auto_offload`. Each worker then plans its own stage's HOT/COLD
layers with its real device.

## Checkpoints

`runtime/checkpoint.py` saves the **logical** state through each tensor's
authoritative copy, keyed by tensor id. That is never just what happens to be on
the GPU. A checkpoint loads under any residency policy, and optimizer state is
recreated on the tier the loading policy uses. Tests resume static ↔ offload in
both directions and match an uninterrupted run bit for bit (CPU).

## MPS (Apple Silicon)

MPS memory is unified: "offloading" to RAM does not free physical memory the way
it does on a discrete GPU. It only moves the bytes out of the Metal working
set that `recommended_max_memory` bounds. The MPS adapter uses synchronous
copies and synchronizes before releasing evicted tensors, because PyTorch
documents no cross-stream semantics for MPS. **Nothing about MPS offload has
been measured yet.** Whether a cheaper zero-copy path exists on unified memory
should be measured, not assumed.

## Observability

* Per step (`metrics.jsonl`, `record["residency"]`): prefetch count/hits/misses/late,
  evictions, writebacks, H2D/D2H bytes and ms, stall and synchronous-load
  time, cache hit ratio, accelerator/RAM resident and peak, pinned bytes,
  budget, headroom, violations. Also `memory_transfer_s`,
  `memory_overlapped_s`, `exposed_memory_transfer_s` from the timeline.
* Timeline (Perfetto): `TENSOR_PREFETCH` (measured on the transfer stream),
  `TENSOR_LOAD`, `PREFETCH_STALL`, `TENSOR_EVICT`, `TENSOR_WRITEBACK`,
  `OPTIMIZER_OFFLOAD`, `TENSOR_ACCELERATOR_RESIDENT` (one lane per layer), next to
  compute and network spans.
* CLI: `meshtrain tensors list|inspect CONFIG`, `meshtrain inspect tensors CONFIG`,
  `meshtrain memory plan CONFIG`, `meshtrain memory status [--run DIR]`,
  `meshtrain benchmark offload`, `meshtrain experiment offload-correctness`,
  `meshtrain experiment capacity --memory-strategy ...`.

## Limits of this phase

* Layer granularity only. A single layer larger than the budget cannot train.
* `after_use` reloads every COLD layer 2·M times per step. Fewer microbatches
  mean less traffic.
* Activations are not offloaded (V1.5 behaviour). Recompute is not implemented.
* Host RAM becomes the capacity limit: about 8 bytes per offloaded parameter with
  the accelerator optimizer, about 16 with `cpu_offload` (pinned masters and
  gradients plus pageable AdamW state).
* Remote RAM (V2.5) is not implemented.

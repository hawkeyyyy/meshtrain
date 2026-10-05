# MeshTrain V1 Architecture

MeshTrain V1 is a **synchronous pipeline-parallel training runtime** for
heterogeneous machines. A sequential model is cut into contiguous *stages*;
each stage runs on one worker (CUDA, MPS or CPU). Activations travel forward
and activation-gradients travel backward over **explicit network boundaries**.
No collective library (NCCL/Gloo) is involved, so CUDA and MPS machines can
participate in the same pipeline.

Terminology (deliberately precise): MeshTrain provides *heterogeneous
distributed model execution* with *explicit tensor placement*. The aggregate
accelerator memory of the cluster can hold a model larger than any single
device, but every kernel still reads only its own local device memory. This is
**not** shared VRAM.

```
            control plane (HTTP/JSON, FastAPI)
   ┌──────────────────────────────────────────────────────┐
   │                    Coordinator                       │
   │  registry · heartbeats · planner · jobs · metrics    │
   └───────▲───────────────▲───────────────▲──────────────┘
           │ register/      │               │ long-poll commands,
           │ heartbeat      │               │ report metrics
   ┌───────┴──────┐  ┌──────┴───────┐  ┌────┴─────────┐
   │ Worker A     │  │ Worker B     │  │ Worker C     │
   │ CUDA stage 0 │══▶ CUDA stage 1 │══▶ MPS stage 2  │   data plane (raw TCP,
   │              │◀══              │◀══ + loss       │   binary TensorPackets)
   └──────────────┘  └──────────────┘  └──────────────┘
        ══▶ FORWARD_ACTIVATION / TARGET     ◀══ BACKWARD_GRADIENT
```

## Components

| Package | Responsibility |
|---|---|
| `meshtrain.coordinator` | FastAPI app: worker registry, heartbeat/offline tracking, job state, planning + stage assignment, command queue, metrics sink. **Never runs PyTorch training.** |
| `meshtrain.worker` | Worker agent: hardware detection, registration, heartbeat thread, command loop, data-plane listener, stage executor. `device.py` holds the device adapters. |
| `meshtrain.runtime` | Model execution independent of transport: `Stage`, distributed-autograd boundary, GPipe microbatch schedule, `TensorPacket`, tensor (de)serialization. |
| `meshtrain.networking` | Wire protocol (`protocol.py`: framing, validation), `Transport` interface (`transport.py`) with `PipeTransport` (same machine, separate processes) and `TCPTransport`/`TCPListener` (`tcp.py`); HTTP control client (`control.py`). Knows nothing about CUDA/MPS. |
| `meshtrain.profiler` | Hardware detection, compute benchmark (`compute_score`), worker-to-worker latency/bandwidth probes. |
| `meshtrain.planner` | Layer graph profiles (param/activation bytes, FLOPs), cost model, contiguous partition planner, memory accounting. |
| `meshtrain.models` | Deterministic test models expressed as an ordered list of layers (MLP, tiny Transformer). |
| `meshtrain.experiments` | Reproducible experiment drivers (correctness, placement, capacity, transformer benchmark, recording of cluster runs) that update `docs/v1-results.md`. `emulation.py` provides emulated heterogeneous CPU workers for hosts without a real cluster. |
| `meshtrain.config`, `telemetry`, `summary` | Validated YAML config (pydantic), structured event logging, `metrics.jsonl` and run summaries. |
| `meshtrain.future` | Placeholder interfaces only (see below). |

Layering rules: the runtime talks to a `Transport` (bytes in, bytes out) and
a `DeviceAdapter` (move/synchronize/memory). Transports never see
`torch.device`; CUDA/MPS specifics live only in `worker/device.py`.

## Coordinator

* `POST /workers/register` (requires `X-MeshTrain-Token`) stores hardware
  metadata (platform, CPU, RAM, accelerators with backend/name/memory), the
  data-plane address and capabilities.
* `POST /workers/{id}/heartbeat` refreshes liveness. A monitor thread marks a
  worker `OFFLINE` after `heartbeat_timeout_s`. If that worker belongs to a
  running job, the job transitions to `FAILED` with a message naming the
  worker, and `STOP_JOB` is queued to the surviving workers.
* Commands are delivered by **long polling** (`GET /workers/{id}/commands`).
  It is simpler than WebSockets, works through any HTTP stack and is enough
  for V1 control traffic.
* `POST /jobs` validates a config, plans the placement (see planner), and
  queues one `START_STAGE` command per stage.
* Workers post per-step metrics to `POST /jobs/{id}/metrics`; the coordinator
  appends them to `runs/<job-id>/metrics.jsonl`.

## Worker

`meshtrain worker join HOST:PORT` detects hardware, picks a device adapter
(`cuda` → `mps` → `cpu`, or forced with `--device`), starts its data-plane
TCP listener, registers, heartbeats, then executes commands:
`RUN_BENCHMARK`, `PROBE_NETWORK`, `START_STAGE`, `STOP_JOB`.

`START_STAGE` instantiates **only the stage's own layers** (each layer is
seeded independently, so a stage's parameters are identical to the same layers
of a single-process model), connects to the downstream stage's listener and
accepts the upstream stage's connection, then runs the pipeline loop.

## Explicit distributed-autograd boundary

PyTorch's autograd graph never spans machines. Each stage owns a local graph;
the runtime stitches them together by hand.

```
# ---- forward, stage k, microbatch m ----
if k == 0:
    x = batch_inputs[m]                      # local data, no grad needed
else:
    t = recv(FORWARD_ACTIVATION, step, m)    # bytes -> CPU tensor
    x = t.to(device).detach().requires_grad_(True)   # boundary leaf
h = module_k(x)                              # local autograd graph
ctx[step, m] = (x, h)                        # keep graph alive
if k == last:
    loss = criterion(h, target[m]) / M       # mean over microbatches
    loss.backward()                          # local backward
    send(BACKWARD_GRADIENT, x.grad, step, m) # to stage k-1
    del ctx[step, m]
else:
    send(FORWARD_ACTIVATION, h.detach(), step, m)   # to stage k+1

# ---- backward, stage k < last, microbatch m ----
g = recv(BACKWARD_GRADIENT, step, m).to(device)
x, h = ctx.pop((step, m))                   # the *saved* graph for m
h.backward(g)                                # dL/dparams_k and dL/dx
if k > 0:
    send(BACKWARD_GRADIENT, x.grad, step, m)

# ---- end of step, every stage independently ----
after all M backward passes of this step:
    optimizer_k.step(); optimizer_k.zero_grad()
```

Correctness argument: by the chain rule, `dL/dθ_k = (dh_k/dθ_k)^T · dL/dh_k`.
`h.backward(g)` computes exactly that product given `g = dL/dh_k`, and stage
k+1 supplies `g` as `x_{k+1}.grad`, which equals `dL/dh_k` because
`x_{k+1}` is a numerically identical copy of `h_k`. Dividing each
microbatch loss by `M` and accumulating `.grad` over microbatches makes the
summed gradient equal to the full-batch mean-loss gradient (equal-size
microbatches). `tests/integration/test_gradient_equivalence.py` checks this
against single-process PyTorch.

Contexts are keyed by `(step_id, microbatch_id)`; a gradient for an unknown
key is a protocol error (never silently applied to another microbatch). A
stage refuses to run its optimizer step while contexts are still pending.

## Pipeline schedule (GPipe-style)

Stage 0 splits the batch into `M` microbatches and issues all `M` forwards
(fill), then waits for `M` gradients (drain). Every other stage is
event-driven: a receiver thread per link pushes decoded packets into one inbox,
and the compute thread handles them in arrival order:

* `FORWARD_ACTIVATION` → forward (last stage: forward + loss + backward
  immediately, i.e. it never holds more than one microbatch graph)
* `TARGET` → relayed downstream by middle stages, consumed by the last stage
* `BACKWARD_GRADIENT` → backward, send gradient upstream
* after `M` backward passes → optimizer step → next step

Targets are produced by stage 0 (the only stage that owns the data loader)
and relayed forward. Losses travel back to stage 0 inside the `meta` of the
gradient packets, so the driver can print the training curve.

The compute thread is single-threaded per stage, so step `s+1` activations
are only processed after the optimizer step of step `s`: training is fully
synchronous (no stale weights).

## Control plane vs. data plane

| | Control plane | Data plane |
|---|---|---|
| Transport | HTTP/JSON (FastAPI + httpx) | raw TCP, length-prefixed binary frames |
| Traffic | registration, heartbeats, commands, status, metrics, benchmark results | activations, gradients, targets, network probes |
| Size | small JSON | header ≤ 64 KiB JSON + raw tensor bytes (bounded) |

Tensors are never sent through JSON.

## Tensor lifecycle

```
CREATE → LOCAL_DEVICE → DETACH → CPU_STAGING → SERIALIZE → NETWORK
       → DESERIALIZE → TARGET_DEVICE → COMPUTE → GRADIENT → reverse path
```

Each phase is timed (`runtime/serialization.py::TensorLifecycle`) and
logged/recorded: `detach`, `to_cpu` (includes device synchronization),
`serialize`, `network_send`, `network_recv`, `deserialize`,
`to_device`. Device→CPU copies use `DeviceAdapter.synchronize()` around
the copy so timings are honest on asynchronous backends.

## Device adapters

`DeviceAdapter` (`move_tensor`, `synchronize`, `memory_stats`,
`supports(op)`) with `CPUDeviceAdapter`, `CUDADeviceAdapter` and
`MPSDeviceAdapter`. Capability queries let the runtime refuse unsupported
work explicitly (e.g. `float64` on MPS) instead of failing deep inside a
kernel. New backends (ROCm, XPU) only need a new adapter.

## Planner and cost model

See `planner/cost.py` for the documented formulas:

```
transfer_time(bytes, i→j) = latency[i][j] + bytes / bandwidth[i][j]
compute_time[v][i]        = flops[v] * (1 + bwd_factor) / measured_flops[i]
stage_time[s]             = Σ compute_time + recv_comm + send_comm
pipeline_step_time        ≈ (M + S − 1) · max_s(stage_time[s] / M)
```

Memory per stage = params + grads + optimizer state + saved activations
(× microbatches in flight) + temporary allowance. The `auto` strategy runs a
dynamic program over contiguous layer ranges and worker orderings minimizing
the bottleneck stage time subject to every worker's memory budget, using
boundary activation size as a tie-breaker. `equal` and `compute` strategies
exist for comparison (Experiment 4).

## Memory accounting

`planner/memory.py` tracks, per stage: parameters, gradients (same size, persistent across
microbatches), optimizer state (0x for SGD, 2x for Adam/AdamW), saved activations (boundary tensors
plus the bytes autograd saves for backward, measured on the `meta` device with
`saved_tensors_hooks`; times the number of microbatches in flight: M for all but the last stage,
1 for the last stage; times a 1.25 safety factor), and a temporary allowance (2x the largest layer
output). The reserved headroom is `max(15% of total, 0.5 GB)`. MPS budgets use
`torch.mps.recommended_max_memory()` because its memory is unified with the OS. At runtime every
stage reports the same categories from the tensors it actually holds (`Stage.memory_report`, plus
peak saved bytes counted with the same hooks), along with the device allocator's view
(`DeviceAdapter.memory_stats`). Example (`MemoryEstimate.format`):

```
Worker: rtx8
    total accelerator memory          8.00 GB
    reserved (headroom)               1.20 GB
    parameters                        1.10 GB
    gradients                         1.10 GB
    optimizer state                   2.20 GB
    saved activations (est.)          0.40 GB
    temporary (est.)                  0.05 GB
```

## Metrics and logs

Every event line carries timestamp, worker, job, step, microbatch and event name. Workers write
`runs/<job>/events-<worker>.jsonl`. The coordinator appends every stage's per-step record (loss,
step/forward/backward/communication/idle time, bytes sent and received, lifecycle phase timings,
memory, utilization, samples/s) to `runs/<job>/metrics.jsonl`. At the end of a job it writes
`summary.json` and `summary.txt`. Pass `trace: true` in a `START_STAGE` command (or
`log_microbatch_events`) for per-microbatch `FORWARD_COMPLETE` / `TRANSFER` / `BACKWARD_COMPLETE`
lines.

## Failure handling

No recovery in V1, but no hangs: every blocking receive has a timeout
(`network.timeout_s`), peer disconnects raise immediately, the coordinator
marks silent workers offline and fails their job, and errors propagate to the
CLI with the failing worker's message.

## Future hooks (interfaces only)

`meshtrain/future.py` defines `MemoryTier`, `TensorStore`,
`PlacementPolicy`, `RecomputePolicy`, `CompressionPolicy` and
`MigrationPolicy` as small abstract interfaces. Nothing implements them in V1.

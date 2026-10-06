# Apple MPS workers

An Apple Silicon Mac joins like any other worker:

```sh
uv run meshtrain join <code>            # picks MPS automatically; --device mps to force
```

## How MPS participates

MPS stages exchange tensors only through MeshTrain's explicit boundaries. There is no NCCL and no shared
communicator:

```
forward:   CUDA → pinned host → TCP → host buffer → MPS
backward:  MPS → host → TCP → pinned host → CUDA
```

`MPSDeviceAdapter` (`worker/device.py`) provides:

| | |
|---|---|
| detection | `torch.backends.mps.is_available()` |
| transfer | `begin_d2h` / `begin_h2d`: **synchronous** copies on the compute thread (see below) |
| synchronisation | `torch.mps.synchronize()` |
| memory | `recommended_max_memory()` as the device total (unified memory: the OS shares it); available = total − driver-allocated |
| capabilities | probed at startup: fp16, bf16 (depends on the macOS version), float64 never, plus per-op probes (`matmul`, `sdpa`, `layer_norm`, …) |
| forward, backward, optimizer | standard PyTorch on `mps` |
| benchmarking | the same matmul / MLP / copy-rate benchmark as other backends |

## Why copies are synchronous on MPS

PyTorch exposes no separate MPS transfer stream and documents no cross-stream event semantics for it. V1.5 does
not assume any. Device↔host copies run on the compute thread and are followed by `torch.mps.synchronize()`.
This is correctness over forced asynchrony. MPS stages still overlap **network** sends and receives with
compute, because those run on transport threads.

## Memory planning on MPS

* Default safety factor **0.80**, versus 0.85 for CUDA, because unified memory is shared with macOS and other
  apps. Override with `memory.backend_safety_factor: {mps: ...}`.
* There is no framework reserve (unlike CUDA's context); the recommended working-set size already excludes
  what the OS needs.
* Runtime validation and the startup probe work the same as on CUDA.

## Correctness checks

```sh
meshtrain experiment device-correctness --devices cpu,mps          # on the Mac
meshtrain experiment device-correctness --devices cpu,cpu,mps
```

These compare loss, first-step gradients, per-parameter gradient norms, first-update parameter deltas and output
logits against a single-process CPU reference, with tolerance 5e-3 relative. MPS kernels and reduction orders
differ from CPU, so the check is not bitwise. The same checks are pytest tests marked `@pytest.mark.mps`. They
skip on machines without MPS and **have not been run yet**. A three-machine CUDA → MPS → CPU run over a
physical LAN (Apple Silicon MacBook Air) matched a CPU single-process loss curve within 1.05e-7 relative over
100 steps; see the `cuda-cpu-mps-lan` section of [v1-results.md](v1-results.md).

For the target cluster (RTX → RTX → M2) run `meshtrain train configs/cuda_cuda_mps.yaml`, then
`meshtrain results record runs/<job-id> --section experiment3`. Recording replays the loss curve on CPU and
reports the deviation.

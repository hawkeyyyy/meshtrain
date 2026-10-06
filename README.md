# MeshTrain

MeshTrain is a research prototype for training one neural network across **heterogeneous consumer
machines**, such as NVIDIA GPUs (CUDA), Apple Silicon (MPS) and plain CPUs. It splits the model into
pipeline stages and **explicitly transmits activations forward and activation-gradients backward**
over the network. No NCCL or other shared collective library is involved, so a CUDA box and a
MacBook can sit in the same pipeline.

Research direction: *topology-aware distributed tensor memory and heterogeneous distributed model
execution across consumer accelerators.* V1 is the minimal, correct foundation; V1.5 makes it efficient
(1F1B schedule, async transport, overlap, memory accounting, MPS, topology-aware planning). MeshTrain does
**not** create shared VRAM: each kernel uses only its own device's memory. See
[docs/research-notes.md](docs/research-notes.md).

## Status (V1)

| Milestone | Status |
|---|---|
| 1. Local CPU prototype: stages in separate processes, gradient equivalence | **Verified on CPU.** Gradients match single-process PyTorch: MLP error 0, Transformer ≤ 3e-7 relative |
| 2. TCP transport | **Verified over loopback and a physical LAN**: activations and gradients transmitted between two GPU machines |
| 3. CUDA → CUDA | **Measured on RTX 4070 Laptop → GTX 1050 Ti**: 396M and 475M models completed five steps across both GPUs; five-step loss curves match laptop-only runs within 3.6e-7 absolute |
| 4. CUDA → CUDA → MPS | Implemented (`MPSDeviceAdapter`, MPS-marked tests). **Not run as specified** (needs two CUDA machines); a CUDA → MPS → CPU pipeline was measured, see V1.5 milestone 9 |
| 5. Microbatch (GPipe) pipeline | **Verified on CPU.** Microbatch routing and accumulation equal full-batch gradients |
| 6. Hardware / network profiler | **Verified on CPU workers**: compute benchmark, directional latency/bandwidth matrix |
| 7. Static partition planner | **Verified** (unit tests + emulated experiment): equal / compute / auto (memory + compute + network) |
| 8. Tiny Transformer benchmark | **Verified on CPU**: 3 stages over TCP, loss curve matches single-process to 2e-7 |
| 9. Capacity experiment | **Emulated gain 2.59x; physical mesh trained 475M.** The laptop also trained 475M with a 9.10 GiB allocation peak; a dedicated-VRAM-only capacity gain remains unproven. See the hardware comparison in the results document |

All numbers and their environments are in [docs/v1-results.md](docs/v1-results.md).

## Status (V1.5)

Placement is still static: each stage stays on one worker for the whole job. See
[docs/v1.5-architecture.md](docs/v1.5-architecture.md) and [docs/v1-vs-v1.5.md](docs/v1-vs-v1.5.md).

| Milestone | Status |
|---|---|
| 1. V1 audit and baseline | Done: [docs/v1.5-baseline.md](docs/v1.5-baseline.md) |
| 2. Event/state-machine pipeline (GPipe preserved) | **Verified on CPU** (gradient equivalence) |
| 3. Async transport with bounded queues | **Verified on CPU/loopback**, including back-pressure and a 4-stage × 32-microbatch stress test |
| 4. Buffer reuse, pinned CUDA staging | Buffer reuse **verified on CPU**; pinned/stream path: CUDA-marked tests **pass on an RTX 4070 Laptop** ([v2-baseline](docs/v2-baseline.md)) |
| 5. 1F1B on CPU | **Verified**: gradients equal GPipe and single-process; stage-0 activations 37.9 → 14.2 MB |
| 6. 1F1B on CUDA | CUDA-marked tests **pass on an RTX 4070 Laptop** ([v2-baseline](docs/v2-baseline.md)) |
| 7. Compute/communication overlap | **Measured on CPU** with an emulated 100 Mbit/s link: 864 → 501 ms/step |
| 8. Memory estimation + startup OOM replanning | **Verified on CPU** with emulated device memory; estimator matches the earlier RTX 4070 + 1050 Ti outcomes (396M fits, 475M rejected). It is conservative: a later run trained 475M across both GPUs, a split this estimator still rejects |
| 9. MPS adapter, CUDA→CUDA→MPS | **Measured CUDA → MPS → CPU over a physical LAN** (RTX 4070 Laptop → Apple Silicon MacBook Air → Fedora x86_64 CPU): 100 steps, max relative loss deviation 1.05e-7 vs CPU single-process. MPS-marked pytest tests not yet run |
| 10. Topology-aware planner | **Verified** (unit tests) |
| 11. Timeline tracing, benchmark reports | **Verified on CPU**: Chrome traces, overlap/idle breakdown, prediction accuracy |
| 12. TensorStore / MemoryTier interfaces | Interfaces + stable tensor IDs (implemented in V2, below) |

Defaults changed in V1.5: `pipeline.schedule: 1f1b`, `transport.async: true`, `placement.strategy:
topology_aware`. Set `schedule: gpipe` and `async: false` to get V1 behaviour.

## Status (V2: tensor residency, local-memory phase)

V2 separates **where a layer computes** (unchanged from V1.5) from **where its state lives**. Offloaded
layers keep their weights, gradients and (optionally) AdamW state in host RAM and are loaded onto the
accelerator only while needed. Design: [docs/v2-architecture.md](docs/v2-architecture.md). Baseline:
[docs/v2-baseline.md](docs/v2-baseline.md). Measurements: [docs/v2-results.md](docs/v2-results.md).

| Milestone | Status |
|---|---|
| 1. Freeze and benchmark V1.5 | Done: tag `v1.5-stable`, 245 passed / 7 skipped |
| 2. TensorStore metadata | **Verified**: stable ids, roles, authoritative/cached tiers, dirty, version, per-tier bytes |
| 3. Artificial accelerator budgets | **Verified on CUDA**: ledger enforcement + hard allocator cap (`memory.accelerator_budget*`) |
| 4–5. Local-RAM parameter offload, CPU and CUDA | **Verified**: bit-exact vs static on CPU and CUDA (accelerator optimizer) |
| 6. Multi-step optimizer correctness | **Verified** per step (params, grads, logits); CPU AdamW on CUDA within Adam rounding |
| 7. Eviction and writeback | **Verified**: after_use / after_backward, LRU under pressure, dirty writeback |
| 8–9. Prefetch, overlap with compute | **Measured on RTX 4070 Laptop**: hit ratio 0.73 → 0.98; transfers mostly exposed (PCIe-bound) |
| 10. Optimizer-state offload | **Verified**: AdamW state + masters in RAM, CPU step |
| 11. Automatic residency planner | **Verified**: benefit/byte heuristic within the budget; cluster planner uses it |
| 12. Capacity vs V1.5 | **Measured on RTX 4070 Laptop**: 2 GiB cap 81.4M → 238.8M (2.93×); full 8 GiB GPU 317.5M → 396.3M trains (static OOMs) |
| 13. Documentation and traces | Done; residency spans in Perfetto timelines |
| 14. Remote RAM | V2.5, below |

`memory.strategy: static` (default) is the unchanged V1.5 path.

## Status (V2.5: remote RAM as a backing tier)

Another machine's RAM can hold layers' master weights and AdamW state. Layers are fetched over the network,
staged in local RAM and copied to the GPU before they compute, and every update is written back with a
version check. GPU kernels still read only local VRAM. Design: [docs/v2.5-remote-memory.md](docs/v2.5-remote-memory.md).
Results: [docs/v2.5-results.md](docs/v2.5-results.md).

Measured on an RTX 4070 laptop + a CPU-only Fedora machine over 154 Mbit/s Wi-Fi, with the same 1 GiB GPU
and 1.5 GiB local RAM budgets: largest trainable model **42.1M (static) → 81.4M (local RAM offload) →
160.1M (+3.5 GiB of the Fedora machine's RAM)**. Losses are bit-identical to a run without remote memory.
Each step took 179 s at 160.1M (3.8 GB over the network per step). This is a capacity tool, not a
throughput tool, on a commodity link.

```sh
# on the RAM machine
meshtrain join CODE --remote-ram-budget-mb 8192        # or: meshtrain tensor-server --budget-mb 8192
# on the GPU machine
meshtrain remote status --probe
meshtrain experiment capacity --memory-strategy static --memory-strategy local-offload-reuse     --memory-strategy remote-offload --budget-mb 1024 --local-ram-budget-mb 1536     --remote-ram-budget-mb 3584 --remote-worker fedora
```

Trusted private networks only: the cluster token authenticates, but there is no encryption.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Clone the repo on every machine, then:

```sh
uv sync
uv run pytest            # CUDA/MPS tests skip automatically when the device is absent
```

PyTorch wheels:

* **Linux + NVIDIA**: the default PyPI wheel includes CUDA.
* **macOS (Apple Silicon)**: the default wheel supports MPS.
* **Windows + NVIDIA**: PyPI's default wheel is CPU-only. After `uv sync`, install a CUDA build and
  stop uv from reverting it:

  ```powershell
  uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
  $env:UV_NO_SYNC = "1"      # so `uv run` keeps the CUDA build
  ```

Check what a machine sees:
`uv run python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`

Open TCP **8080** (coordinator, HTTP) on the first machine and **29500** (data plane) on every
machine. All machines must reach each other on a private network.

## Quick start: multiple machines

On the first machine:

```sh
uv run meshtrain start
```

It starts the coordinator plus a worker for this machine, then prints a join command such as:

```
meshtrain join k3f9x2mq7d@192.168.1.20
```

Run that command on every other machine (prefix it with `uv run` if the venv isn't activated):

```sh
uv run meshtrain join k3f9x2mq7d@192.168.1.20
```

Then, from any machine in the cluster (the cluster is remembered in `~/.meshtrain/cluster.json`, so no
flags are needed):

```sh
uv run meshtrain status
uv run meshtrain benchmark                          # compute scores + bandwidth/latency matrix
uv run meshtrain train configs/cuda_cuda_mps.yaml
```

The part before `@` is the cluster token, generated by `start`. Treat the join command like a
password. Each worker picks CUDA, then MPS, then CPU; force one with `--device cuda|mps|cpu`. A GPU
with less than 1 GB of memory (e.g. a 128 MB legacy card) is reported but treated as unusable, so
that machine joins as a CPU helper. If a machine has several network adapters or a VPN and picks the
wrong address, pass `--advertise-host <LAN IP>` to `start` or `join`.

`train` prints the plan (`Stage 0 → …`), then loss, step time, communication time and bytes per
step. It ends with a per-stage summary. Metrics go to `runs/<job-id>/metrics.jsonl` on the
first machine. Record a run in `docs/v1-results.md` with
`meshtrain results record runs/<job-id> --section experiment2`. If a worker dies, the coordinator
marks it `OFFLINE` and fails the job with a message naming it. Nothing hangs: every network wait has
a timeout.

The long forms still work: `meshtrain coordinator start`, `meshtrain worker join HOST:PORT --token T`,
`meshtrain cluster status`, `meshtrain cluster benchmark`, `meshtrain plan CONFIG`.

## Quick start: one machine

```sh
# every stage as a local process, TCP between them (no coordinator)
uv run meshtrain train configs/local_cpu.yaml --local

# full control/data-plane path with a coordinator and 3 CPU workers on this host
MESHTRAIN_TOKEN=dev uv run python scripts/local_cluster.py --workers 3 --port 8090
MESHTRAIN_TOKEN=dev uv run meshtrain --coordinator 127.0.0.1:8090 cluster benchmark
MESHTRAIN_TOKEN=dev uv run meshtrain --coordinator 127.0.0.1:8090 train configs/tiny_transformer_cpu.yaml
```

## Interactive terminal

```sh
uv run meshtrain console
```

The console groups the cluster, training, and monitoring tools. Enter commands directly at the
`meshtrain >` prompt, for example `status`, `jobs`, `plan configs/local_cpu.yaml`, or
`train configs/local_cpu.yaml`. Use `help train` for command options, `help` for the tools menu,
and `exit` to leave. It uses the remembered cluster and accepts `--coordinator` / `--token` overrides.
Blank input does not repeat a command; help and command errors return to the prompt.

Cluster startup and the console display a colored MESHTRAIN ASCII banner. Status, benchmark, and
job tables use terminal colors. Redirected output stays plain; set `NO_COLOR`
to disable colors in a terminal. `meshtrain jobs` shows unique measured steps, latest loss, and
the latest first-stage step time. Hardware capacity experiments resolve flags, environment variables,
and the remembered cluster in that order, just like other cluster commands.

## Live training dashboard

With a cluster running, open a second terminal on this machine:

```sh
uv run meshtrain dashboard
```

Open **http://127.0.0.1:8081** to see workers, training runs, the live loss curve, and per-stage
compute, communication, and memory metrics. Select a run, inspect chart points with the mouse or
arrow keys, choose a linear/log scale, pause updates, or export a run as JSON. Updates arrive every
two seconds. The dashboard uses the remembered cluster; `--coordinator` and `--token` also work.
Use `--port 8091` if port 8081 is occupied.

The dashboard binds to this machine's loopback interface and keeps the cluster token on the server.
It observes jobs launched through the coordinator (`meshtrain train CONFIG`); `--local` runs are not
included. Run history is available while the coordinator process is running.

## Experiments

```sh
uv run meshtrain experiment correctness --transport tcp   # Exp 1: single process vs distributed gradients
uv run meshtrain experiment transformer                   # Milestone 8 benchmark
uv run meshtrain experiment placement                     # Exp 4 on emulated heterogeneous CPU workers
uv run meshtrain experiment placement --cluster cluster.json   # Exp 4 predictions for a real cluster
uv run meshtrain experiment capacity                      # Exp 5, emulated budgets
uv run meshtrain experiment capacity --mode hardware      # Exp 5 on a real cluster (via coordinator)
uv run meshtrain experiment device-correctness --devices cuda,cuda   # per-device gradient check vs CPU
uv run meshtrain benchmark pipeline                       # V1 vs V1.5 modes, local, emulated link
uv run meshtrain benchmark pipeline --cluster --config configs/two_cuda.yaml   # same on a real cluster
uv run meshtrain inspect placement configs/tiny_transformer_cpu.yaml           # planned stages, memory, comm
```

Each cluster run writes `runs/<job>/timeline.json` (open in https://ui.perfetto.dev),
`timeline_report.txt` and `prediction_accuracy.json`. See [docs/performance.md](docs/performance.md).

Experiments 2 and 3 are `meshtrain train configs/two_cuda.yaml` / `configs/cuda_cuda_mps.yaml`
followed by `meshtrain results record`. Recording replays the first steps on CPU and reports the
loss-curve deviation as a correctness check.

Hardware capacity comparisons require at least two online workers. Each mesh trial uses one stage
per worker; the report includes the stage count. CUDA Adam/AdamW plans budget an additional
parameter set for foreach update workspace and cap usable VRAM using the latest heartbeat's free
memory plus unused allocator cache. Cached pools larger than physical VRAM are excluded from that
cap. These are estimates; a dedicated-VRAM-only baseline still requires verifying the GPU driver's
Sysmem Fallback configuration before testing.

## Configuration

```yaml
job:       {name: tiny-transformer-test, seed: 0}
model:     {type: tiny_transformer, layers: 12, hidden_size: 512, heads: 8, vocab_size: 1024, seq_len: 128}
training:  {batch_size: 16, microbatch_size: 4, learning_rate: 0.0003, optimizer: adamw, steps: 500}
placement: {strategy: topology_aware, num_stages: 3}  # topology_aware (=auto) | equal | compute | manual
workers:   {allow: [cuda, mps, cpu]}
network:   {tensor_transport: tcp, timeout_s: 120}
pipeline:  {schedule: 1f1b, max_inflight_microbatches: null}   # 1f1b | gpipe
transport: {async: true, pinned_memory: true, buffer_pool: true}
memory:    {safety_factor: 0.85, backend_safety_factor: {mps: 0.80}, max_replans: 2}
# V2 (optional): keep layer state in RAM and load it onto the GPU only when needed
# memory:  {strategy: auto_offload, accelerator_budget: 6GB, optimizer_offload: true, prefetch_distance: 1}
```

V2 commands: `meshtrain memory plan CONFIG` (which layers stay on the GPU), `meshtrain tensors list CONFIG`,
`meshtrain memory status`, `meshtrain benchmark offload`, `meshtrain experiment offload-correctness`,
`meshtrain experiment capacity --memory-strategy static --memory-strategy auto-offload --budget-mb 2048`.

Memory settings are explained in [docs/memory-accounting.md](docs/memory-accounting.md), schedules in
[docs/pipeline-scheduling.md](docs/pipeline-scheduling.md), MPS in [docs/mps.md](docs/mps.md).

Configs are validated strictly: unknown keys, bad sizes and non-contiguous manual stages are
rejected. See `src/meshtrain/config.py`.

## How it works

```
Stage A (CUDA)  ──h1 (bytes over TCP)──▶  Stage B (CUDA)  ──h2──▶  Stage C (MPS) + loss
       ◀──────────────── dL/dh1 ───────────────    ◀──── dL/dh2 ────
```

Each stage runs local autograd on its own layers. A received activation becomes a fresh leaf
(`detach().requires_grad_(True)`). Its `.grad` is sent upstream, where `h.backward(grad)` continues
the chain rule. Details are in [docs/architecture.md](docs/architecture.md); the wire formats are in
[docs/protocol.md](docs/protocol.md).

## Security

V1 is meant for **trusted machines on a private network**. It uses a shared cluster token for
registration and data-plane connections, and validates message sizes and formats. It never unpickles
or executes data received from peers. There is **no TLS**, and the token travels in clear text. **Do
not expose MeshTrain to the Internet or to untrusted peers.**

## Repository layout

```
src/meshtrain/
  coordinator/  server.py registry.py scheduler.py state.py      control plane (FastAPI)
  worker/       worker.py device.py executor.py heartbeat.py      worker agent, device adapters
                capabilities.py                                    probed op/dtype support
  runtime/      stage.py pipeline.py distributed_autograd.py      execution (transport-independent)
                scheduler.py outbox.py buffers.py timeline.py      V1.5: schedules, async sends, buffers, traces
                memory_check.py trace_report.py tensor_store.py
                tensor_packet.py serialization.py local.py
                offload.py checkpoint.py                           V2: residency manager, checkpoints
  networking/   protocol.py transport.py tcp.py control.py         data/control plane I/O
                emulation.py                                       bandwidth/latency emulation (experiments)
  profiler/     hardware.py benchmark.py network.py
  planner/      graph.py partition.py cost.py memory.py residency.py
  models/       mlp_test.py tiny_transformer.py
  experiments/  correctness, placement, capacity, transformer, record, emulation
  cli.py config.py telemetry.py summary.py future.py
configs/  scripts/  tests/{unit,integration,distributed}/  docs/
```

The package lives under `src/meshtrain/` (src layout) rather than the top-level `meshtrain/` shown in
the original sketch.

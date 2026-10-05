# MeshTrain

MeshTrain is a research prototype for training one neural network across **heterogeneous consumer
machines**, such as NVIDIA GPUs (CUDA), Apple Silicon (MPS) and plain CPUs. It splits the model into
pipeline stages and **explicitly transmits activations forward and activation-gradients backward**
over the network. No NCCL or other shared collective library is involved, so a CUDA box and a
MacBook can sit in the same pipeline.

Research direction: *topology-aware distributed tensor memory and heterogeneous distributed model
execution across consumer accelerators.* V1 is the minimal, correct foundation. MeshTrain does
**not** create shared VRAM: each kernel uses only its own device's memory. See
[docs/research-notes.md](docs/research-notes.md).

## Status (V1)

| Milestone | Status |
|---|---|
| 1. Local CPU prototype: stages in separate processes, gradient equivalence | **Verified on CPU.** Gradients match single-process PyTorch: MLP error 0, Transformer ≤ 3e-7 relative |
| 2. TCP transport | **Verified over loopback TCP on one host.** Not yet run across two physical machines |
| 3. CUDA → CUDA | Implemented (`CUDADeviceAdapter`, CUDA-marked tests). **Not run: no CUDA hardware was available** |
| 4. CUDA → CUDA → MPS | Implemented (`MPSDeviceAdapter`, MPS-marked tests). **Not run: no CUDA/MPS hardware was available** |
| 5. Microbatch (GPipe) pipeline | **Verified on CPU.** Microbatch routing and accumulation equal full-batch gradients |
| 6. Hardware / network profiler | **Verified on CPU workers**: compute benchmark, directional latency/bandwidth matrix |
| 7. Static partition planner | **Verified** (unit tests + emulated experiment): equal / compute / auto (memory + compute + network) |
| 8. Tiny Transformer benchmark | **Verified on CPU**: 3 stages over TCP, loss curve matches single-process to 2e-7 |
| 9. Capacity experiment | **Emulated on CPU only** (scaled-down budgets): capacity gain 2.59x. The hardware mode exists but has **not** been run on physical GPUs |

All numbers and their environments are in [docs/v1-results.md](docs/v1-results.md).

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

Open TCP **8080** (coordinator, HTTP) on the coordinator machine and **29500** (data plane) on every
worker. All machines must reach each other on a private network.

## Quick start: multiple machines

Pick a shared secret and use it everywhere:

```sh
export MESHTRAIN_TOKEN=change-me              # PowerShell: $env:MESHTRAIN_TOKEN="change-me"
```

Machine A (coordinator + worker):

```sh
uv run meshtrain coordinator start                       # listens on 0.0.0.0:8080
uv run meshtrain worker join 192.168.1.20:8080           # in a second terminal
```

Machine B and the Mac:

```sh
uv run meshtrain worker join 192.168.1.20:8080           # --device cuda|mps|cpu to force a backend
```

A worker picks CUDA, then MPS, then CPU. A GPU with less than 1 GB of memory (e.g. a 128 MB legacy
card) is reported but treated as unusable, so that machine joins as a CPU helper. If the
auto-detected address is wrong (VPNs, several NICs), pass `--advertise-host <LAN IP>`.

Then, from any machine (`--coordinator 192.168.1.20:8080` or `MESHTRAIN_COORDINATOR`):

```sh
uv run meshtrain cluster status
uv run meshtrain cluster benchmark --output cluster.json   # compute scores + bandwidth/latency matrix
uv run meshtrain plan configs/cuda_cuda_mps.yaml           # dry-run placement
uv run meshtrain train configs/cuda_cuda_mps.yaml
uv run meshtrain results record runs/<job-id> --section experiment3   # add the run to docs/v1-results.md
```

`train` prints the plan (`Stage 0 → …`), then loss, step time, communication time and bytes per
step. It ends with a per-stage summary. Metrics go to `runs/<job-id>/metrics.jsonl` on the
coordinator. If a worker dies, the coordinator marks it `OFFLINE` and fails the job with a message
naming it. Nothing hangs: every network wait has a timeout.

## Quick start: one machine

```sh
# every stage as a local process, TCP between them (no coordinator)
uv run meshtrain train configs/local_cpu.yaml --local

# full control/data-plane path with a coordinator and 3 CPU workers on this host
MESHTRAIN_TOKEN=dev uv run python scripts/local_cluster.py --workers 3 --port 8090
MESHTRAIN_TOKEN=dev uv run meshtrain --coordinator 127.0.0.1:8090 cluster benchmark
MESHTRAIN_TOKEN=dev uv run meshtrain --coordinator 127.0.0.1:8090 train configs/tiny_transformer_cpu.yaml
```

## Experiments

```sh
uv run meshtrain experiment correctness --transport tcp   # Exp 1: single process vs distributed gradients
uv run meshtrain experiment transformer                   # Milestone 8 benchmark
uv run meshtrain experiment placement                     # Exp 4 on emulated heterogeneous CPU workers
uv run meshtrain experiment placement --cluster cluster.json   # Exp 4 predictions for a real cluster
uv run meshtrain experiment capacity                      # Exp 5, emulated budgets
uv run meshtrain experiment capacity --mode hardware      # Exp 5 on a real cluster (via coordinator)
```

Experiments 2 and 3 are `meshtrain train configs/two_cuda.yaml` / `configs/cuda_cuda_mps.yaml`
followed by `meshtrain results record`. Recording replays the first steps on CPU and reports the
loss-curve deviation as a correctness check.

## Configuration

```yaml
job:       {name: tiny-transformer-test, seed: 0}
model:     {type: tiny_transformer, layers: 12, hidden_size: 512, heads: 8, vocab_size: 1024, seq_len: 128}
training:  {batch_size: 16, microbatch_size: 4, learning_rate: 0.0003, optimizer: adamw, steps: 500}
placement: {strategy: auto, num_stages: 3}     # auto | equal | compute | manual (+ stages: [...])
workers:   {allow: [cuda, mps, cpu]}
network:   {tensor_transport: tcp, timeout_s: 120}
```

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
  runtime/      stage.py pipeline.py distributed_autograd.py      execution (transport-independent)
                tensor_packet.py serialization.py local.py
  networking/   protocol.py transport.py tcp.py control.py         data/control plane I/O
  profiler/     hardware.py benchmark.py network.py
  planner/      graph.py partition.py cost.py memory.py
  models/       mlp_test.py tiny_transformer.py
  experiments/  correctness, placement, capacity, transformer, record, emulation
  cli.py config.py telemetry.py summary.py future.py
configs/  scripts/  tests/{unit,integration,distributed}/  docs/
```

The package lives under `src/meshtrain/` (src layout) rather than the top-level `meshtrain/` shown in
the original sketch.

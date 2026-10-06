# V2 baseline: MeshTrain V1.5 frozen

Everything on this page was measured on the V1.5 code before any V2 change,
from a clean checkout of the baseline commit.

## Identity

| | |
|---|---|
| Commit | `ceee374d8945c66c3c5a39e589903833f0bdcf60` (`main`, "Record first CUDA -> MPS -> CPU LAN run") |
| Tag | `v1.5-stable` (annotated, local) |
| Select V1.5 behaviour in V2 | `memory.strategy: static` and `optimizer.execution: accelerator` (both are the defaults) |

## Environment

| | |
|---|---|
| Machine | `hawkey`: Windows 11 (10.0.26200) laptop, 22 logical CPUs, 15.4 GiB RAM |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB (8,585,216,000 bytes), driver 610.62 |
| Software | Python 3.12, torch 2.6.0+cu124 |
| Apple MPS | Not on this machine. Measured once over the LAN (see below). |

## Tests

`pytest` on the clean V1.5 checkout: **245 passed, 7 skipped** (202 s).
The CUDA-marked tests ran on the RTX 4070. The skips:

* 5 × no Apple MPS device (`test_hardware_pipelines.py`, `test_device.py`)
* 1 × "CUDA present" (a CPU-only fallback test, `test_device.py`)
* 1 × MPS adapter unit test

## MPS result

There is no MPS on this machine. The one MPS measurement is the three-machine
LAN run recorded in [v1-results.md](v1-results.md#cuda-cpu-mps-lan). It ran RTX 4070
(Windows) → MacBook Air MPS → Fedora CPU for 100 steps, with max relative loss
deviation **1.05e-7** vs CPU single-process. The MPS-marked pytest tests have
never run.

## Pipeline behaviour (V1.5)

* Each stage owns a contiguous layer range on one worker for the whole job.
  All of its tensors (parameters, gradients, AdamW state, activations) live on
  that worker's device.
* Defaults: 1F1B schedule, asynchronous transport with bounded queues, pinned
  CUDA staging, topology-aware static placement, startup memory validation and
  OOM replanning.
* Distributed autograd crosses stage boundaries explicitly (activations forward,
  activation-gradients backward over TCP).

## Throughput and peak memory (single device, one stage, no network)

Reproduce with `python scripts/v15_baseline.py throughput --device cuda`.
AdamW, batch 16, 4 microbatches, 12 steps (first excluded), uncapped allocator.

| Model | Parameters | ms/step | samples/s | Peak allocated |
|---|---|---|---|---|
| tiny_transformer 12 × 512, seq 128, vocab 1024 | 38.9M | 120.4 | 132.9 | 0.96 GiB |
| tiny_transformer 24 × 1024, seq 128, vocab 1024 | 304.5M | 3371.7 | 4.7 | 6.47 GiB |

The 304M row is about 28× slower per parameter than the capacity rows below.
It runs near the physical limit with no cap. The Windows driver can then spill
into shared system memory, so its time is not a clean compute number.

## Maximum model size (static placement, single device)

Reproduce with `python scripts/v15_baseline.py capacity --device cuda --cap-gb N`.
tiny_transformer, hidden 1280, 8 heads, seq 64, vocab 1024; AdamW, batch 8,
4 microbatches, 3 steps. The allocator is **hard-capped** with
`torch.cuda.set_per_process_memory_fraction`. This also stops the driver from
spilling into shared system memory, so OOM is real. Sizes step by 4 blocks.

| Cap | Largest that trained | Peak allocated | ms/step | First failure |
|---|---|---|---|---|
| 8 GiB (whole GPU) | 317.5M (16 blocks) | 6.28 GiB | 326 | 396.3M (20 blocks): CUDA OOM |
| 2 GiB | 81.4M (4 blocks) | 1.64 GiB | 94 | 160.1M (8 blocks): CUDA OOM |

Measured static step times at cap 8 GiB: 81.4M 103 ms, 160.1M 161 ms,
238.8M 233 ms, 317.5M 326 ms.

For comparison, [v1-results.md](v1-results.md) records the same laptop
training a 475.0M model *without* an allocator cap, at a 9.10 GiB peak. That
exceeds the GPU's 8 GiB, so it relied on the Windows driver's shared-memory
fallback, not VRAM.

## Known bottlenecks (going into V2)

1. **Static model state dominates device memory.** With AdamW, every parameter
   costs about 4 (weights) + 4 (gradient) + 8 (Adam moments) + ~4 (foreach update
   temporaries) bytes on the device for the whole job. This happens whether or
   not its layer is computing. This is the limit V2 attacks.
2. **Network.** Multi-machine runs over Wi-Fi are communication-bound. A 3-machine
   run spent 633–992 ms per step communicating vs 21–567 ms computing per
   stage, with spikes up to 20 s.
3. **GPU benchmark underrates the 4070 about 8×.** It uses a 1024² matmul
   (1.02 vs about 8.6 TFLOP/s measured with 4096²). Placement therefore gives the
   GPU too little work.
4. **CPU stages** are an order of magnitude slower per layer than the GPU.

# MeshTrain V2 results (local-memory phase)

Every number here was measured on the hardware named in its section.
Nothing is extrapolated. Raw data (JSON + CSV, one row per trial) is in
[data/v2/](data/v2/). Each directory is named after the run it came from, and the same names appear
below as `runs/<name>`.

## Environment

| | |
|---|---|
| Machine | `hawkey`: Windows 11 laptop, 15.4 GiB RAM (about 4.5–6.5 GiB free during the runs; other applications and a running MeshTrain worker held the rest) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB, driver 610.62. About 0.85 GiB is used by other processes (desktop, MeshTrain worker). |
| Software | torch 2.6.0+cu124, Python 3.12 |
| Code | branch `claude/meshtrain-v2-memory` |
| Not measured | Apple MPS offload, multi-machine offload jobs |

Common setup unless stated: tiny_transformer (hidden 1280, 8 heads, seq 64, vocab 1024),
AdamW lr 3e-4, batch 8 split into 4 microbatches, 1F1B, one stage on one GPU, 3 steps (first excluded from
timings). Each trial runs in a fresh process. The budget is a **hard CUDA allocator cap**
(`set_per_process_memory_fraction`), so "OOM" below is a real CUDA out-of-memory, and the Windows driver
cannot spill into shared system memory. A host-RAM guard skips trials that would push the machine into
the pagefile. Those are reported as "RAM-limited", not as successes.

Strategies: **A** static (V1.5) · **B** local-offload (layers in RAM, AdamW on GPU, synchronous loads) ·
**C** optimizer-offload (B + AdamW state/masters in RAM, CPU step) · **D** C + prefetch distance 1 ·
**auto** planner-chosen residency with CPU AdamW · **auto-g** planner-chosen residency with AdamW on GPU.

## Correctness

### Per-step equivalence (`tests/integration/test_offload.py`, `meshtrain experiment offload-correctness`)

tiny_transformer 4 × 64, 4 AdamW steps, 4 microbatches. Every step compares loss, accumulated
gradients, updated parameters and the logits of a fixed probe batch with static training.

| Device | Policy | Max abs diff (loss / grads / params / logits) |
|---|---|---|
| CPU | manual offload (prefetch 0, 1, 2), after_backward, keep-resident mix, optimizer offload, both | **0 / 0 / 0 / 0** (bit-exact, every step) |
| RTX 4070 | manual offload (prefetch 0, 1), after_backward, keep-resident mix (AdamW on GPU) | **0 / 0 / 0 / 0** (bit-exact, every step) |
| RTX 4070 | optimizer offload (CPU AdamW) | 1.2e-7 / 1.5e-8 / 5.4e-5 / 6.0e-7 after 4 steps; mean param diff 1.1e-8 |

The one non-zero case is explained, not tolerated blindly. With identical gradients, CPU and CUDA AdamW
differ by at most 2.4e-7 after 4 steps. The larger parameter difference sits entirely on elements whose
Adam second moment is √v ≈ 4e-12 ≪ eps = 1e-8. The worst tensor is `qkv.bias`, the attention key bias,
whose true gradient is zero because softmax ignores it. There the update is driven by rounding noise in
either implementation. 61 of 192 elements of that tensor differ by more than 1e-6, with median √v there
5e-12 vs 1.6e-5 overall.

### Capacity-scale runs (loss after 3 steps)

| Model | Static (A) | Offload, AdamW on GPU (B / auto-g) | Offload, CPU AdamW (C / D / auto) |
|---|---|---|---|
| 81.4M | 7.024250268936157 | 7.024250268936157 (identical) | 7.024250507354736 (Δ 2.4e-7) |
| 160.1M | 7.07812237739563 (uncapped, baseline) | 7.07812237739563 (identical) | 7.078122138977051 (Δ 2.4e-7) |
| 238.8M | 7.161629676818848 (uncapped, baseline) | n/a | 7.161629676818848 (auto) |
| 317.5M | 7.170265078544617 | 7.170265078544617 (identical, B and auto-g) | n/a |
| 396.3M | CUDA OOM | 7.223023533821106 (B and auto-g identical) | n/a |

Other checks: checkpoints are residency-independent. Saving under one policy and resuming under another
(static ↔ offload, either optimizer) reproduces an uninterrupted run bit for bit on CPU. A two-process TCP
pipeline with offload in both stages matches static bit for bit.

## Capacity: largest trainable model

### 2 GiB budget (`runs/offload-capacity-20261006-122104`)

`meshtrain experiment capacity --memory-strategy ... --budget-mb 2048 --device cuda`

| Strategy | 81.4M | 160.1M | 238.8M | 317.5M | P_max | Capacity gain |
|---|---|---|---|---|---|---|
| A static | 108 ms, 1.69 GB | **CUDA OOM** | | | 81.4M | 1.00× |
| B local-offload | 702 ms, 1.01 GB | 1719 ms, 1.68 GB | **CUDA OOM** | | 160.1M | 1.97× |
| C optimizer-offload | 700 ms, 0.31 GB | 1616 ms, 0.34 GB | RAM-limited | | ≥160.1M | ≥1.97× |
| D C + prefetch | 879 ms, 0.40 GB | 1458 ms, 0.43 GB | RAM-limited | | ≥160.1M | ≥1.97× |
| auto | 254 ms, 0.74 GB | 482 ms, 1.40 GB | 1047 ms, 1.85 GB | RAM-limited | **≥238.8M** | **≥2.93×** |

Cells show ms/step and peak reserved GPU memory. Every successful trial stayed within the 2 GiB cap.
"RAM-limited" means the trial was skipped because it would have needed more host RAM than was free
(C/D/auto need about 16 B/parameter of RAM). These are lower bounds on capacity, not GPU limits: C and D
held only 0.34–0.43 GB of the GPU at 160M.

### Full GPU, 8 GiB budget (`runs/offload-capacity-20261006-122229`, `runs/offload-capacity-8g-pageable-20261006-122600`)

| Strategy | 317.5M | 396.3M | 475.0M |
|---|---|---|---|
| A static | 366 ms/step, 6.42 GB | **CUDA OOM** | |
| B local-offload | 3280 ms, 2.97 GB | 6366 ms, 3.63 GB ‡ | RAM-limited |
| auto-g (AdamW on GPU) | 306 ms, 6.31 GB (all layers resident) | **1868 ms, 7.27 GB** ‡ (18 resident / 4 offloaded) | RAM-limited |

‡ run with pageable host memory (`pin_host_memory: false`). The first 396.3M offload attempts, with
pinned memory and only about 2–5 GB of RAM free, failed with CUDA OOM even though only 3.3 GB of the GPU
was allocated. On Windows, exhausted *host* memory surfaces this way. With about 6.3 GB free, the same
396.3M B run trains with pinned memory too (4.37 s/step vs 6.06 s/step pageable, uncapped allocator).

**V2 success criterion met on the same hardware:** 396.3M fails under static V1.5 placement
with a real CUDA OOM on the full 8 GiB GPU, and trains with local-RAM offload: P_v2_max ≥ 396.3M >
P_v1.5_max = 317.5M (≥ 1.25×). Under a 2 GiB budget the gain is ≥ 2.93× (81.4M → 238.8M), limited by host RAM.

### Throughput cost

| Comparison | Static | Offload | Slowdown |
|---|---|---|---|
| 81.4M at 2 GiB, auto (all layers fit, CPU AdamW) | 108 ms | 254 ms | 2.36× |
| 81.4M at 2 GiB, B | 108 ms | 702 ms | 6.5× |
| 317.5M at 8 GiB, B | 366 ms | 3280 ms | 9.0× |
| 317.5M at 8 GiB, auto-g (all resident) | 366 ms | 306 ms | 0.84× ¹ |
| Each strategy at its own P_max (2 GiB): A 81.4M vs auto 238.8M | 9.29 steps/s | 0.96 steps/s | 9.7× |

¹ auto-g kept every layer resident at 317.5M, so it is static with a different optimizer call pattern.
The static run peaked at 6.42 GB, near the cap. Its extra time is plausibly allocator pressure, but that
was not isolated.

## Where the time goes (2 GiB, 160.1M; `results.json` fields)

| Strategy | Step | Blocking loads | Prefetch stall | Compute | Optimizer | H2D per step | Effective H2D |
|---|---|---|---|---|---|---|---|
| B | 1719 ms | 679 ms | – | 214 ms | 424 ms | 5.94 GB | 9.5 GB/s |
| C | 1616 ms | 593 ms | – | 479 ms | 243 ms | 4.75 GB | 8.7 GB/s |
| D | 1458 ms | 3 ms | 246 ms | 440 ms | 235 ms | 4.76 GB | 9.9 GB/s |
| auto | 482 ms | 0 | 0 | 353 ms | 363 ms | 0.60 GB | – |

* `after_use` eviction reloads every offloaded layer 2 × 4 microbatches = 8 times per step. That is
  4.75 GB of PCIe traffic for 160M parameters. Transfers, not compute, dominate. Fewer microbatches or a
  layer-major schedule would cut this. Neither is implemented.
* Prefetch (D vs C) removes almost all *blocking* loads, but 246 ms of stall remains. The compute it can
  overlap with (440 ms over 10 layers × 8 passes, about 5.5 ms per layer pass) is shorter than loading
  one 78 MB block (about 8 ms at ~10 GB/s). The step is transfer-bound, and overlap can hide at most the
  compute time. At 81.4M prefetch was *slower* (879 vs 700 ms/step). Larger is not automatically better.
* The planner-chosen residency (auto) is 3× faster than offloading everything at the same budget, because
  it keeps resident whatever fits.

## Memory budget vs step time (`runs/offload-budget-sweep-20261006-122814`)

`meshtrain benchmark offload --blocks 8 --sweep budget`: 160.1M parameters, auto-offload (CPU AdamW, prefetch 1),
batch 32 in 8 microbatches (CLI defaults), 4 steps.

| Budget | ms/step | samples/s | Peak reserved | H2D per step | Exposed transfer |
|---|---|---|---|---|---|
| 1.0 GB | 2017 | 15.9 | 0.80 GB | 7.19 GB | 1381 ms |
| 1.5 GB | 1123 | 28.5 | 1.27 GB | 3.90 GB | 687 ms |
| 2.0 GB | 635 | 50.4 | 1.45 GB | 0.60 GB | 114 ms |
| 3.0 GB | 615 | 52.1 | 1.45 GB | 0.60 GB | 116 ms |
| 4.0 GB | 633 | 50.6 | 1.45 GB | 0.60 GB | 127 ms |
| 6.0 GB | 631 | 50.7 | 1.45 GB | 0.60 GB | 118 ms |

From 2 GB up every layer is resident. The remaining 0.6 GB/step of H2D is the CPU optimizer refreshing
device copies. Below that, the planner trades memory for transfer smoothly: 1 GB costs 3.2× the time of 2 GB.

## Prefetch distance vs throughput (`runs/offload-prefetch-sweep-20261006-122723`)

`meshtrain benchmark offload --blocks 8 --budget-mb 2048 --sweep prefetch`: 160.1M, strategy D (all layers
offloaded, CPU AdamW), batch 32 in 8 microbatches, 4 steps.

| Distance | ms/step | Peak reserved | Exposed transfer |
|---|---|---|---|
| 0 (synchronous) | 3153 | 0.41 GB | 2502 ms |
| 1 | 3015 | 0.50 GB | 2373 ms |
| 2 | 2930 | 0.57 GB | 2285 ms |
| 3 | 3080 | 0.73 GB | 2410 ms |

Each step of distance costs about one layer of device memory. The best distance (2) is only 7% faster than
synchronous. With 16 reloads per layer per step (9.5 GB of PCIe traffic), the step is transfer-bound.
Prefetching cannot hide more than the compute time, and distance 3 is worse again.

## What is not measured

* Apple MPS offload (no Mac on this machine). The adapter path is synchronous and untested on hardware.
* Offload inside multi-machine jobs. The coordinator plans with it and workers build stages with it (unit
  and loopback-pipeline tests), but no LAN run has used it yet.
* Capacity beyond the host-RAM limit of this session (about 6 GB free). The 2 GiB-budget offload strategies
  stopped on RAM, not on the GPU.

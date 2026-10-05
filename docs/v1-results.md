# MeshTrain V1 Results

This file is updated automatically by the experiment drivers
(`meshtrain experiment ...`, see README). Each section states the hardware it
was produced on. **Sections marked "not yet run" have no measurements**:
in particular, nothing here was measured on physical CUDA or MPS hardware
unless the section's environment line says so.

## Summary

| Question | Answer so far | Evidence |
|---|---|---|
| Do distributed gradients equal single-process gradients? | Yes on CPU: MLP exactly (0 error), Transformer to ~1e-7 relative | Experiment 1 (pipe and TCP), test suite |
| Does distributed training follow the single-process trajectory? | Yes on CPU (loss curves agree to ~1e-7) | Milestone 8, cluster demo |
| Do all stages update and does the loss decrease? | Yes (2 and 3 stages, MLP and Transformer) | tests, Milestone 8, cluster demo |
| CUDA → CUDA, CUDA → CUDA → MPS | **Not measured**: no physical CUDA/MPS hardware was available | Experiments 2 and 3 (pending) |
| Capacity gain (primary metric) | 2.59x **in emulation** (scaled-down budgets, CPU processes); unmeasured on real devices | Experiment 5 |
| Does the cost model rank placements correctly? | Not reliably: on the emulated workers the measured ranking differed from the prediction | Experiment 4 |

## Experiment 1 — correctness (single process vs. distributed CPU stages)

### Pipe transport (separate processes, same machine — Milestone 1)

<!-- BEGIN:experiment1-pipe -->
_Generated 2026-10-05 13:03 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

Device: CPU only (each stage in its own OS process; tensors cross process boundaries as serialized TensorPackets). Batch split identically in the single-process reference.

| case | transport | stages | microbatches | params | max abs err | max relative err | loss abs diff | result |
|---|---|---|---|---|---|---|---|---|
| MLP 2 stages | pipe | 2 | 1 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| MLP 3 stages | pipe | 3 | 1 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| MLP 2 stages, 4 microbatches | pipe | 2 | 4 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| Transformer (4 blocks, d=64) 3 stages, 2 microbatches | pipe | 3 | 2 | 54 | 3.73e-09 | 1.78e-07 | 0.00e+00 | PASS |

- **MLP 2 stages** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00
- **MLP 3 stages** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00, stage2: 0.00e+00
- **MLP 2 stages, 4 microbatches** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00
- **Transformer (4 blocks, d=64) 3 stages, 2 microbatches** — per-stage max abs error: stage0: 9.31e-10, stage1: 9.31e-10, stage2: 3.73e-09
<!-- END:experiment1-pipe -->

### TCP transport (separate processes, loopback TCP — Milestone 2)

The stages talk over real TCP sockets. In this run both ends were on one host (127.0.0.1); no
multi-machine run has been recorded yet.

<!-- BEGIN:experiment1-tcp -->
_Generated 2026-10-05 13:04 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

Device: CPU only (each stage in its own OS process; tensors cross process boundaries as serialized TensorPackets). Batch split identically in the single-process reference.

| case | transport | stages | microbatches | params | max abs err | max relative err | loss abs diff | result |
|---|---|---|---|---|---|---|---|---|
| MLP 2 stages | tcp | 2 | 1 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| MLP 3 stages | tcp | 3 | 1 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| MLP 2 stages, 4 microbatches | tcp | 2 | 4 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| Transformer (4 blocks, d=64) 3 stages, 2 microbatches | tcp | 3 | 2 | 54 | 3.73e-09 | 1.78e-07 | 0.00e+00 | PASS |

- **MLP 2 stages** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00
- **MLP 3 stages** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00, stage2: 0.00e+00
- **MLP 2 stages, 4 microbatches** — per-stage max abs error: stage0: 0.00e+00, stage1: 0.00e+00
- **Transformer (4 blocks, d=64) 3 stages, 2 microbatches** — per-stage max abs error: stage0: 9.31e-10, stage1: 9.31e-10, stage2: 3.73e-09
<!-- END:experiment1-tcp -->

## Experiment 4 — placement strategies

### Emulated heterogeneous CPU workers (single host)

Interpretation: only `auto` respects every (emulated) memory budget. The two memory-unaware plans
would overflow the small, fast worker. Measured step times on this 4-core host vary by roughly ±15%
between runs, and the predicted ordering of the strategies was **not** confirmed by measurement. The
cost model is first-order (see `planner/cost.py`) and needs real-cluster calibration.

<!-- BEGIN:experiment4-emulated -->
_Generated 2026-10-05 13:04 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

Emulated heterogeneous CPU workers on one host (compute differences are real: separate processes with different torch thread counts, measured by the benchmark; memory budgets are emulated and enforced by the planner). Model: tiny Transformer (8.18M params, 12 layers), AdamW, batch 16, 4 microbatches, 12 steps per plan.

| worker | emulated as | measured GFLOP/s | compute_score |
|---|---|---|---|
| fast-small | 2 threads, 90 MB | 191.3 | 1.00 |
| slow-large | 1 thread, 260 MB | 124.0 | 0.65 |
| slow-mid | 1 thread, 150 MB | 133.7 | 0.70 |

| strategy | placement (worker:layers) | memory-feasible | predicted ms/step | measured ms/step | tensor MB held/budget per stage | loss |
|---|---|---|---|---|---|---|
| equal | fast-small:0-3 / slow-large:4-7 / slow-mid:8-11 | NO: stage 0 on fast-small needs 0.10 GB but only 0.08 GB is usable | 507.2 | 355.1 | 90.5/81 / 118.3/234 / 52.2/135 EXCEEDED | 6.435 -> 5.942 |
| compute | fast-small:0-4 / slow-large:5-7 / slow-mid:8-11 | NO: stage 0 on fast-small needs 0.14 GB but only 0.08 GB is usable | 446.3 | 382.0 | 119.6/81 / 76.0/234 / 52.2/135 EXCEEDED | 6.435 -> 5.942 |
| auto | slow-large:0-4 / fast-small:5-6 / slow-mid:7-11 | yes | 375.4 | 396.8 | 119.6/234 / 60.2/81 / 68.5/135 ok | 6.435 -> 5.942 |

_Plans that violate an emulated memory budget were still executed (a CPU process cannot hit the emulated limit) so their speed can be compared; on a real accelerator they would be rejected by the planner or fail with out-of-memory._

_Held memory = parameters + gradients + optimizer state + peak saved activations (boundary tensors + tensors autograd saved for backward), measured from what each stage actually held. Network links are loopback (not emulated), so the network-aware part of `auto` has little to optimise here._
<!-- END:experiment4-emulated -->

## Experiment 5 — capacity (primary V1 metric)

### Emulated (scaled-down budgets on CPU processes)

<!-- BEGIN:experiment5-emulated -->
_Generated 2026-10-05 13:05 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

**Mode: emulated.** Target cluster memory scaled down 32x and emulated by single-threaded CPU processes on one host; memory limits are the planner's accounting (params + grads + AdamW state + saved activations + temporary, 10% headroom) against the emulated budget, *not* real device OOMs. The largest models were then actually trained and the tensor bytes each stage held were measured. Model family: tiny Transformer (vocab 1024, seq 64, 8 heads), AdamW, batch 8, 4 microbatches. Parameter counts are as built (not scaled).

| worker | real memory | emulated budget | measured GFLOP/s |
|---|---|---|---|
| rtx8 | 8.0 GB | 256 MB | 140.0 |
| rtx12 | 12.0 GB | 384 MB | 137.6 |
| m2 | 10.7 GB | 341 MB | 99.3 |

| model (blocks x hidden) | params | rtx8 alone | rtx12 alone | m2 alone | mesh (auto) | mesh total required |
|---|---|---|---|---|---|---|
| 2x128 | 0.7M | fits | fits | fits | fits (1 stages) | 15 MB |
| 4x192 | 2.2M | fits | fits | fits | fits (1 stages) | 45 MB |
| 4x256 | 3.7M | fits | fits | fits | fits (1 stages) | 70 MB |
| 6x256 | 5.3M | fits | fits | fits | fits (2 stages) | 129 MB |
| 6x320 | 8.1M | fits | fits | fits | fits (3 stages) | 194 MB |
| 6x384 | 11.5M | fits | fits | fits | fits (3 stages) | 257 MB |
| 8x384 | 15.0M | - | fits | fits | fits (3 stages) | 346 MB |
| 8x448 | 20.3M | - | - | - | fits (3 stages) | 443 MB |
| 8x512 | 26.3M | - | - | - | fits (2 stages) | 516 MB |
| 10x512 | 32.6M | - | - | - | fits (3 stages) | 659 MB |
| 12x512 | 38.9M | - | - | - | fits (3 stages) | 784 MB |
| 12x576 | 49.1M | - | - | - | - | - |

|  | P_max | device(s) | steps/s (measured) | tensor MB held/budget | loss (first -> last) |
|---|---|---|---|---|---|
| single device | 15.0M | rtx12 | 1.362 | rtx12: 255/346 MB | 7.101 -> 6.925 |
| MeshTrain | 38.9M | m2 -> rtx8 -> rtx12 | 1.087 | m2: 270/307 MB, rtx8: 197/230 MB, rtx12: 271/346 MB | 7.086 -> 6.810 |
| MeshTrain, same model as single | 15.0M | rtx8 -> rtx12 -> m2 | 2.639 | rtx8: 167/230 MB, rtx12: 81/346 MB, m2: 68/307 MB | 7.101 -> 6.925 |

**capacity_gain = P_mesh_max / max(P_single_max) = 38.9M / 15.0M = 2.59x**  
**throughput penalty = 1.362 / 1.087 steps/s = 1.25x slower**  
Same model (15.0M) alone vs on the mesh: 1.362 vs 2.639 steps/s

_Caveat: these throughput numbers are **not representative of a real cluster**. Each emulated worker is a single-threaded process, so the 3-stage mesh uses 3 CPU cores where the single device uses 1, and links are loopback TCP (no real network cost). On real GPUs over a LAN the mesh is expected to be much slower than a single device; only the capacity ratio is meaningful here._

_5 steps per run; loss changes over so few steps are not a convergence result. Real-hardware capacity numbers require `meshtrain experiment capacity --mode hardware` on the physical cluster; none have been recorded yet._
<!-- END:experiment5-emulated -->

### Physical hardware

<!-- BEGIN:experiment5-hardware -->
_not yet run — requires the physical CUDA/MPS cluster (`meshtrain experiment capacity --mode hardware`)._
<!-- END:experiment5-hardware -->

## Milestone 8 — tiny Transformer benchmark

<!-- BEGIN:milestone8-transformer -->
_Generated 2026-10-05 13:04 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

CPU only, 3 stage processes over loopback TCP. Model: 1.23M params (8 layers: embedding, 6 blocks, LM head), stages [(0, 3), (3, 5), (5, 8)], batch 16 = 4 microbatches of 4, AdamW lr 1e-3, synthetic arithmetic-sequence next-token task.

- One-step gradient equivalence: max relative error 2.74e-07 (3.ln1.bias), max abs error 2.79e-09 over 78 parameter tensors

- Training: loss 5.0327 -> 0.3597 over 150 steps; parameters changed on every stage: True; max relative deviation from single-process loss curve: 2.35e-07

- Throughput: 61.2 ms/step, 266.0 samples/s, 1.062 MB sent per step

| step | MeshTrain loss | single-process loss |
|---|---|---|
| 0 | 5.0327 | 5.0327 |
| 25 | 3.0978 | 3.0978 |
| 50 | 2.1333 | 2.1333 |
| 100 | 0.9873 | 0.9873 |
| 149 | 0.3597 | 0.3597 |

| stage | worker | fwd ms | bwd ms | comm ms | idle ms | MB sent/step | params MB | saved act peak MB |
|---|---|---|---|---|---|---|---|---|
| 0 | local0 | 13.9 | 20.4 | 1.4 | 19.9 | 0.268 | 1.67 | 9.22 |
| 1 | local1 | 14.1 | 19.4 | 2.3 | 19.5 | 0.531 | 1.59 | 9.47 |
| 2 | local2 | 13.5 | 19.8 | 1.2 | 21.9 | 0.263 | 1.65 | 2.50 |
<!-- END:milestone8-transformer -->

## Cluster path on one host (coordinator + 3 CPU worker processes)

This exercises the full control and data plane: HTTP registration, heartbeats, planning, TCP links,
and per-step metrics reporting. All three workers are CPU processes on one machine. Their names are
labels only, not real devices.

<!-- BEGIN:cluster-cpu-demo -->
Run `tiny-transformer-cpu-20261005-130007-fe03` — hardware: alpha (cpu, x86_64) -> beta (cpu, x86_64) -> gamma (cpu, x86_64)

Model: tiny_transformer 1.2M params (8 layers); batch 16, 4 microbatches, adamw.

| steps | loss first -> last | ms/step | steps/s | samples/s | MB/step on network | max rel. loss deviation vs CPU single-process (first 20 steps) |
|---|---|---|---|---|---|---|
| 150 | 5.0327 -> 0.3597 | 60.6 | 16.49 | 267.0 | 1.06 | 1.00e-07 |

| stage | worker | backend | layers | planned GB (need / usable) | device peak GB | fwd ms | bwd ms | comm ms | idle ms | MB sent/step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | alpha | cpu | 0-1 | 0.01 / 12.82 | 0.73 | 6.8 | 10.7 | 1.4 | 37.6 | 0.27 |
| 1 | beta | cpu | 2-4 | 0.03 / 12.82 | 0.74 | 19.3 | 26.8 | 2.2 | 5.4 | 0.53 |
| 2 | gamma | cpu | 5-7 | 0.01 / 12.82 | 0.73 | 13.0 | 19.0 | 1.2 | 23.1 | 0.26 |
<!-- END:cluster-cpu-demo -->

## Experiment 2 — CUDA pipeline (RTX 8 GB + RTX 12 GB)

<!-- BEGIN:experiment2 -->
_not yet run — no physical CUDA hardware was available. Run `meshtrain train configs/two_cuda.yaml`,
then `meshtrain results record runs/<job-id> --section experiment2`._
<!-- END:experiment2 -->

## Experiment 3 — heterogeneous pipeline (RTX 8 GB + RTX 12 GB + Apple M2)

<!-- BEGIN:experiment3 -->
_not yet run — no physical CUDA/MPS hardware was available. Run `meshtrain train configs/cuda_cuda_mps.yaml`,
then `meshtrain results record runs/<job-id> --section experiment3`._
<!-- END:experiment3 -->

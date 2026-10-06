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
| CUDA → CUDA | **Measured**: 396M and 475M models completed five steps on RTX 4070 Laptop → GTX 1050 Ti over a physical LAN | Hardware comparison, 2026-10-06 |
| CUDA → CUDA → MPS | **Not measured** | Experiment 3 (pending) |
| Capacity gain (primary metric) | 2.59x **in emulation**; 1.00x for the tested physical sizes with the current driver settings; dedicated-VRAM-only gain unproven | Experiment 5 and hardware comparison |
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
_Generated 2026-10-05 23:38 on hawkey (Windows AMD64, torch 2.6.0+cu124, CUDA available: True, MPS available: False)._

**Mode: hardware** (real devices via the coordinator).

| device | P_max | steps/s at P_max | first failure |
|---|---|---|---|
| hawkey | 475.0M | 0.117 | stage 0 on hawkey: RuntimeError: CUDA error: out of memory CUDA kernel errors might be asynchronously reported at some o |
| server | 0.0M | 0.000 | stage 0 on server: OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 3.94 |
| MeshTrain | 396.3M | 0.144 | stage 1 on server: OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 3.94 |

capacity_gain = 0.83x, throughput penalty = 0.82x
<!-- END:experiment5-hardware -->

### Hardware evidence review (2026-10-05, before the corrected comparison)

The generated table above predates the planner correction. Its "MeshTrain" row does not establish
that the successful model was split across multiple GPUs: the old hardware experiment let `auto`
choose a single worker. The saved artifacts show:

- `capacity-20261005-233740-04d2`: 396.3M parameters, three completed steps, one CUDA stage on
  `hawkey`, all layers `[0, 22)`. No server stage or network boundary was used.
- `capacity-20261005-233804-9b96`: laptop alone completed three steps at 475.0M parameters;
  recorded peak torch allocation was 9.10 GiB against 8.00 GiB device capacity. This is not a
  dedicated-VRAM-only baseline. The artifacts do not record the NVIDIA Sysmem Fallback setting.
- `capacity-20261005-233833-8720`: the 475.0M attempt used `hawkey` layers `[0, 15)` and
  `server` layers `[15, 26)`, but did not complete. The generated report records an OOM on the server.
- `capacity-20261005-233343-70ba`: the earlier server-only sweep completed 143.4M parameters
  in three steps. The later sweep started at 203.7M and failed on the server immediately; its
  reported `0.0M` means no success in that sweep, not that the server has zero capacity.

These trials do **not** demonstrate a completed 396.3M pipeline across both GPUs or a model
larger than either device can train alone. The old estimate counted parameters, gradients and
AdamW state, but omitted CUDA foreach optimizer intermediates (approximately another parameter
set). The corrected planner includes that workspace and caps budgets using reported free VRAM
plus unused allocator cache. Hardware capacity trials now request one stage per online worker
and report the actual stage count at the largest successful tested size.

With the recorded GPU totals and benchmark rates, a corrected dry plan places the 475.0M model
on `hawkey` layers `[0, 17)` and `server` layers `[17, 26)`: estimated requirements are
6.71 / 6.80 GiB and 3.06 / 3.35 GiB respectively. The old server allocation now estimates
3.82 GiB, exceeding its 3.35 GiB budget. The 553.7M candidate has no feasible partition at
the default headroom. These dry plans assume available memory does not further reduce either
budget; live heartbeats may lower them. The V1.5 estimator (CUDA safety factor 0.85, 0.4 GB
framework reserve) is more conservative and rejects this split: it estimates 3.07 GiB on `server`
against a 2.95 GiB usable budget.

New plans remain estimates, not measured training results. A fair capacity comparison needs
fresh single-device and multi-device runs with the same model, dtype, optimizer and batch settings,
with the laptop's dedicated-VRAM-only configuration verified. Aggregate VRAM does not by itself
guarantee a 550–600M model will fit, because optimizer workspace, activations, headroom and
indivisible layer sizes also constrain the partition.

## Corrected hardware comparison — 2026-10-06

Laptop: Windows AMD64, Python 3.12, torch 2.6.0+cu124, RTX 4070 Laptop GPU with 7.996 GiB.
Server: a separate machine on the LAN with GTX 1050 Ti, 3.937 GiB; remote torch version was
not reported. All runs used CUDA. The server-only trials started late on 2026-10-05; the
large comparisons completed on 2026-10-06 (Asia/Katmandu).

Five steps per trial: FP32 tiny Transformer, vocab 1024, sequence length 64, eight attention
heads, AdamW at 3e-4, batch 8 as four microbatches of 2. The large comparisons restarted the
laptop worker between trials. The corrected auto planner forced two stages for mesh runs.
Rates exclude the first warmup step and initialization. GPU peaks below are recorded
per-process CUDA allocation peaks, not measurements of dedicated versus shared residency.

| Target | Parameters | Steps | Steps/s | GPU allocation peaks (GiB) | Run |
|---|---|---|---|---|---|
| Server alone | 153.3M | 5 | 0.782 | server: 2.87 | `capacity-20261005-235818-d450` |
| Laptop alone | 396.3M | 5 | 0.026 | laptop: 7.60 | `capacity-cold-20261006-000710-56aa` |
| Laptop alone | 475.0M | 5 | 0.122 | laptop: 9.10 | `capacity-cold-20261006-001035-0628` |
| Both GPUs | 396.3M | 5 | 0.626 | laptop: 4.56, server: 3.66 | `capacity-cold-20261006-001951-8e7c` |
| Both GPUs | 475.0M | 5 | 0.615 | laptop: 6.07, server: 3.66 | `capacity-cold-20261006-001608-615c` |

The 475M pipeline placed layers `[0, 17)` on the laptop and `[17, 26)` on the server.
Both stages reported gradients, AdamW state and nonzero optimizer time, with five completed
updates. Each pipeline step sent approximately 5.25 MB across the network. The five-step loss
curves at both 396M and 475M matched their laptop-only counterparts within 3.58e-7 absolute.
This verifies actual distributed training at these sizes, not merely a proposed partition.

At the same 475M size, the mesh was **5.02x faster in this five-step comparison** (0.615 versus
0.122 steps/s). This is under the user's current driver configuration: Sysmem Fallback was
left unchanged, without verifying the driver profile. The laptop's 9.10 GiB allocation peak
exceeds its 8 GiB physical capacity. Both the laptop and mesh completed 475M, so capacity gain
for the tested sizes is **1.00x**, with no demonstrated dedicated-VRAM-only capacity gain.
The 396M laptop rate was slower than its 475M rate, illustrating the variability of these
short runs with memory pressure; these are not general performance or convergence results.

The server's largest successful tested model was 153.3M; 178.5M failed with CUDA OOM.
The 553.7M two-GPU candidate was rejected by the planner, not tested to hardware OOM.
The largest successful tested sizes are not exact capacity limits.

Initial setup attempts suffered missed laptop heartbeats, blocked sandboxed outbound LAN
traffic, or an assignment racing the previous worker's pending command poll during restart.
Those attempts are excluded from successful comparisons. Final mesh runs used LAN access,
a 120-second connection window and a 120-second coordinator heartbeat timeout; the final
restart also allowed the old command poll to expire. No fresh compute/network benchmark
was run after coordinator restart, so these rates describe the selected plans rather than
an optimized placement sweep.

The consolidated local evidence is in `runs/capacity-validation-20261006/report.md` and
`comparison.json`. Source manifests preserve failed and rejected attempts as well as the
successful jobs, their plans, loss curves and per-stage memory/timing metrics.

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

<!-- BEGIN:cuda-cpu-mps-lan -->
Run `cuda-cpu-mps-20261006-113414-5d1d` — hardware: hawkey (cuda, NVIDIA GeForce RTX 4070 Laptop GPU) -> Sohans-MacBook-Air.local (mps, Apple arm64 (MPS)) -> fedora (cpu, x86_64)

Model: tiny_transformer 40.0M params (14 layers); batch 16, 4 microbatches, adamw.

| steps | loss first -> last | ms/step | steps/s | samples/s | MB/step on network | max rel. loss deviation vs CPU single-process (first 100 steps) |
|---|---|---|---|---|---|---|
| 100 | 7.7928 -> 2.7797 | 1427.2 | 0.70 | 11.6 | 16.32 | 1.05e-07 |

_device peak = torch allocator peak (CUDA), driver allocation (MPS), process RSS (CPU)._

| stage | worker | backend | layers | planned GB (need / usable) | device peak GB | fwd ms | bwd ms | comm ms | idle ms | MB sent/step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | hawkey | cuda | 0-0 | 0.03 / 6.87 | 0.06 | 9.0 | 12.4 | 633.4 | 772.3 | 4.21 |
| 1 | Sohans-MacBook-Air.local | mps | 1-10 | 1.18 / 4.58 | 1.09 | 119.2 | 220.7 | 992.0 | 226.0 | 7.91 |
| 2 | fedora | cpu | 11-13 | 0.24 / 7.84 | 1.48 | 223.8 | 343.1 | 452.6 | 468.9 | 4.20 |
<!-- END:cuda-cpu-mps-lan -->

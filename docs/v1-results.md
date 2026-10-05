# MeshTrain V1 Results

This file is updated automatically by the experiment drivers
(`meshtrain experiment ...`, see README). Each section states the hardware it
was produced on. **Sections marked "not yet run" have no measurements**:
in particular, nothing here was measured on physical CUDA or MPS hardware
unless the section's environment line says so.

## Experiment 1 — correctness (single process vs. distributed CPU stages)

### Pipe transport (separate processes, same machine — Milestone 1)

<!-- BEGIN:experiment1-pipe -->
_Generated 2026-10-05 12:04 on vm (Linux x86_64, torch 2.14.1+cu130, CUDA available: False, MPS available: False)._

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

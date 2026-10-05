# MeshTrain Research Notes

## Hypothesis

Consumer machines have **fragmented accelerator memory** and **incompatible backends**. A
household or lab might have:

* an 8 GB and a 12 GB NVIDIA GPU (CUDA)
* an Apple M2 laptop (MPS, unified memory)
* an old PC whose GPU is too small to matter (a CPU helper)

No collective library spans these devices: NCCL cannot reach MPS. No single device can hold the
training state of a model larger than about its own memory divided by ~16 bytes per parameter (fp32
weights + grads + Adam).

MeshTrain asks: **can explicit graph partitioning plus explicit tensor transmission turn this
fragmented hardware into a usable distributed training resource?** Specifically, can it train models
whose memory requirements exceed the usable memory of every participating device? What does it
cost in throughput?

Terminology: MeshTrain gives *aggregate accelerator memory* through *explicit tensor placement* and
*heterogeneous distributed model execution*. It does not create shared VRAM. Every kernel reads only
its own device's memory, and tensors cross devices only as explicit network messages. "Virtual
VRAM" is at most an informal metaphor.

## What V1 proves (when its tests and experiments pass)

1. **Correctness of explicit autograd boundaries.** A model cut into stages that exchange detached
   activations forward and activation-gradients backward produces the same parameter gradients as
   single-process PyTorch. With equal microbatches, accumulating `loss/M` gives the full-batch
   gradient. This is checked to fp32 summation-order precision on CPU (Experiment 1, tests).
2. **Backend independence of the boundary.** The transport only moves bytes, plus dtype and shape.
   Any device that can do `tensor.to("cpu")` and back can join a pipeline. Stages need not share a
   backend, an OS, or a collective library.
3. **Synchronous training semantics.** With per-stage optimizer steps after all M backward passes,
   the distributed training trajectory matches single-process training step for step.
4. **Capacity bookkeeping.** The planner's memory accounting (params, grads, optimizer state,
   saved activations, temporary) predicts what each stage holds, so a plan can be checked against
   per-device budgets before launch.

## What V1 does not prove

* **Nothing about real CUDA/MPS performance or capacity yet.** No measurements on physical CUDA or
  MPS hardware are recorded in this repository so far (see `docs/v1-results.md`). The CUDA/MPS
  adapter tests are written but have been skipped on every machine used so far.
* **Real network costs.** All recorded runs use loopback TCP on one host. Real LAN or Wi-Fi
  bandwidth and latency will dominate step time for large activations.
* **Accurate cost prediction.** The cost model ignores compute/communication overlap, allocator
  behaviour and kernel efficiency at small sizes. In the emulated placement experiment the
  predicted ranking of strategies did not match the measured ranking. Treat it as a
  first-order heuristic.
* **Efficiency.** GPipe-style fill/drain leaves stages idle `(S-1)/(M+S-1)` of the time.
  Transfers are synchronous and staged through CPU memory, with no overlap, pinned memory,
  compression or 1F1B schedule.
* **Fault tolerance.** Any worker failure ends the job cleanly. Nothing is recovered.
* **Large real models.** The tiny Transformer and MLP are stand-ins. Hugging Face models are a
  later step.

## Expected trade-off

MeshTrain should be *slower* than a single device whenever the model fits on one device. Pipeline
bubbles, CPU staging and network transfers all cost time. The interesting regime is when the model
does **not** fit on any single device. Then the comparison is "slow" vs "impossible" (or vs
CPU/offload training). The V1 primary metric captures this:

```
capacity_gain      = P_mesh_max / max_i P_single_max[i]
throughput_penalty = steps/s(best single device at its P_max) / steps/s(mesh at its P_max)
```

## Next research steps (post-V1, not implemented)

* Real-hardware runs of Experiments 2, 3 and 5.
* Overlap communication with compute, use pinned staging buffers, and a 1F1B schedule.
* Compress activations/gradients on slow links (`CompressionPolicy`).
* Activation recomputation to trade compute for memory (`RecomputePolicy`).
* Memory tiers (`MemoryTier`, `TensorStore`) for offloading optimizer state to RAM/NVMe.
* Topology-aware placement with measured, asymmetric links (the planner already accepts
  `bandwidth[i][j] != bandwidth[j][i]`).

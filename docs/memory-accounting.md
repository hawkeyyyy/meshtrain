# Memory accounting (V1.5)

## Why it changed

On real hardware (an RTX 4070 Laptop with 8 GB plus a GTX 1050 Ti with 4 GB), V1's planner accepted a
475M-parameter split that then hit CUDA OOM on the 1050 Ti. V1 counted only parameters, gradients, optimizer
state, saved activations and a small temporary against `total − max(15%, 0.5 GB)`. It ignored AdamW's update
temporaries, buffers and the CUDA context.

## The estimate (`planner/memory.py`)

For a stage with parameter bytes P on a device with T bytes:

| Component | Estimate |
|---|---|
| usable budget | `T × safety_factor[backend] − framework_reserve[backend]` |
| parameter_bytes | P |
| gradient_bytes | P |
| optimizer_bytes | 0 (SGD) / 2P (Adam, AdamW) |
| optimizer step temporaries | 0 (SGD) / P (Adam, AdamW multi-tensor update intermediates) |
| master_weight_bytes | 0: V1.5 trains in a single dtype |
| saved_activation_bytes | in-flight × Σ autograd-saved bytes × 1.25 |
| input_buffer_bytes | in-flight × boundary input bytes |
| output_buffer_bytes | in-flight × boundary output bytes + 2 × input-gradient bytes |
| transport_buffer_bytes | CPU backend: 2 × boundary bytes; accelerators: 0 (pinned staging is host RAM, reported separately) |
| temporary_workspace | 2 × largest layer output + largest per-layer saved bytes |

In-flight is 1 for the last stage, M for other stages under GPipe, and `min(S − s, M, max_inflight)` under
1F1B. Autograd-saved bytes come from running each layer on the `meta` device with `saved_tensors_hooks`.

Defaults are configurable under `memory:` and are **starting points, not universal truths**:

```yaml
memory:
  safety_factor: 0.85
  backend_safety_factor: {mps: 0.80}
  framework_reserve_gb: {cuda: 0.4, mps: 0.0, cpu: 0.25}
  validate_runtime_usage: true
  probe_allocation: true
  max_replans: 2
  replan_shrink: 0.85
```

The regression test `test_regression_real_hardware_capacity_outcomes` uses the real outcomes: it accepts the
396M split that trained and rejects the 475M split that OOM'd.

## Runtime validation (`runtime/memory_check.py`)

After a worker materialises its stage, and before connecting to peers:

1. It measures the parameters it holds and the memory still available: CUDA `mem_get_info` plus the allocator
   cache; MPS working set minus driver allocation; CPU available RAM.
2. It records the estimate error (estimated vs actual parameters; on accelerators also the actual allocation
   before training).
3. It raises `StageMemoryError` if the rest of the estimated need clearly exceeds what is available (× safety
   factor).
4. On CUDA/MPS (`probe_allocation`), it allocates the estimated remainder once and frees it. A future mid-step
   OOM becomes a clean startup failure.
5. After the first step it records the actual peak (torch allocator peak on CUDA) against the estimate:
   `memory_validation` in metrics, `prediction_accuracy.json` per run.

Example report line:

```
estimated = 3.82 GB, parameters estimated 1.10 GB vs actual 1.10 GB (+0.0%),
actual_before_training = 1.18 GB (error vs estimated parameters +7.6%)
```

## Startup OOM replanning

```
attempt placement → stage fails validation / probe / OOM before its first step completes
  → worker releases device memory and reports {error, measured capacity, available}
  → coordinator stops the attempt, sets that worker's budget from the measured capacity
    (× replan_shrink), and plans again
  → new attempt (new data-plane id); at most memory.max_replans times
  → otherwise the job fails with the reason
```

Every attempt is in `runs/<job>/placement_attempts.json`. An OOM after training has started is not replanned:
V1.5 has no mid-training migration.

Testing on a CPU-only machine: `MESHTRAIN_EMULATE_DEVICE_MEMORY_GB=<n>` makes a CPU worker report only n GB
available. This is fault injection for tests, and the end-to-end test uses it.

## Reported per step

`memory` in every STEP_COMPLETE record includes:
* parameters, gradients and optimizer state actually held
* peak saved activation bytes and microbatches
* device allocated and peak
* `tensor_store` bytes per role (parameter, gradient, optimizer state, activation)

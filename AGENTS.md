# MeshTrain contributor instructions

- MeshTrain V1 is authorized: a synchronous pipeline-parallel training runtime with explicit
  distributed-autograd boundaries, TCP data plane, FastAPI control plane, profiler and static planner.
- MeshTrain V1.5 is authorized on top of V1, with placement still static: event-driven 1F1B/GPipe
  schedules, async bounded transport, buffer reuse and pinned CUDA staging, compute/communication
  overlap, memory accounting with startup replanning, MPS workers, topology-aware planning, timeline
  tracing, and the `TensorStore`/`MemoryTier` interfaces in `src/meshtrain/runtime/tensor_store.py`.
- Not authorized (V2, only on explicit request): tensor migration or paging, eviction, RAM/NVMe offload,
  remote optimizer state, dynamic repartitioning during training. `LocalTensorStore.evict` must keep
  raising `NotImplementedError` until then.
- Training semantics never change: one `optimizer.step()` per global step, all microbatches on the same
  parameter version. Schedules only reorder work inside a step.
- V1 scope excludes (do not implement unless explicitly requested): shared-memory illusion, remote VRAM
  paging, weight migration, ZeRO/FSDP, optimizer-state sharding, SSD/NVMe paging, async SGD, automatic
  repartitioning, fault-tolerant recovery, internet-scale networking, RL schedulers, Kubernetes, custom
  CUDA kernels. Interfaces for later work live in `src/meshtrain/future.py` as placeholders only.
- Correctness before optimization: any change to the runtime must keep
  `tests/integration/test_gradient_equivalence.py` and `tests/integration/test_1f1b.py` passing.
- Keep transports free of torch device logic; keep CUDA/MPS specifics inside `worker/device.py`.
- Use Python 3.12+ and keep the package under `src/meshtrain/`.
- Prefer uv for dependency management. Run `uv sync` to set up the environment and `uv run pytest` to run tests.
- Hardware-specific tests use the `cuda` / `mps` pytest markers and skip when the device is absent.
- Never claim results for hardware that was not actually used; `docs/v1-results.md` and `docs/v1-vs-v1.5.md`
  sections state their environment, and emulated links are always labelled.
- Keep changes minimal and add dependencies only when required by the task.

# MeshTrain contributor instructions

- MeshTrain V1 is authorized: a synchronous pipeline-parallel training runtime with explicit
  distributed-autograd boundaries, TCP data plane, FastAPI control plane, profiler and static planner.
- V1 scope excludes (do not implement unless explicitly requested): shared-memory illusion, remote VRAM
  paging, weight migration, ZeRO/FSDP, optimizer-state sharding, SSD/NVMe paging, async SGD, automatic
  repartitioning, fault-tolerant recovery, internet-scale networking, RL schedulers, Kubernetes, custom
  CUDA kernels. Interfaces for later work live in `src/meshtrain/future.py` as placeholders only.
- Correctness before optimization: any change to the runtime must keep
  `tests/integration/test_gradient_equivalence.py` passing.
- Keep transports free of torch device logic; keep CUDA/MPS specifics inside `worker/device.py`.
- Use Python 3.12+ and keep the package under `src/meshtrain/`.
- Prefer uv for dependency management. Run `uv sync` to set up the environment and `uv run pytest` to run tests.
- Hardware-specific tests use the `cuda` / `mps` pytest markers and skip when the device is absent.
- Never claim results for hardware that was not actually used; `docs/v1-results.md` sections state their environment.
- Keep changes minimal and add dependencies only when required by the task.

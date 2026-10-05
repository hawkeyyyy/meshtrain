# MeshTrain contributor instructions

- This repository currently contains only project scaffolding.
- Do not implement training, networking, scheduling, model partitioning, or distributed functionality unless explicitly requested.
- Use Python 3.12+ and keep the package under `src/meshtrain/`.
- Prefer uv for dependency management. Run `uv sync` to set up the environment and `uv run pytest` to run tests.
- Keep changes minimal and add dependencies only when required by the task.

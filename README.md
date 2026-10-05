# MeshTrain

MeshTrain is an experimental research runtime for training one neural network across heterogeneous
consumer machines (CUDA GPUs, Apple MPS, CPUs) by splitting it into pipeline stages and explicitly
transmitting activations forward and activation-gradients backward over the network.

Status: **V1 in progress.** See `docs/architecture.md` and `docs/v1-results.md`.

| Milestone | Status |
|---|---|
| 1. Local CPU prototype (separate processes, gradient equivalence) | done, verified on CPU |
| 2. TCP transport | done; verified over loopback TCP on one host (no physical multi-machine run yet) |

## Development

Requires Python 3.12+ and uv.

```sh
uv sync
uv run pytest
```

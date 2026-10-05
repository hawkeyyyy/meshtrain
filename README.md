# MeshTrain

MeshTrain is an experimental distributed training runtime for combining heterogeneous consumer hardware to train and fine-tune models larger than a single device can comfortably handle.

Initial research direction:

- distributed pipeline model parallelism
- heterogeneous GPUs/computers
- activation and gradient transfer between machines
- LoRA/QLoRA training
- topology-aware scheduling
- later support for RAM/NVMe layer streaming

Status: early research project.

## Development

Requires Python 3.12+ and uv.

```sh
uv sync
uv run pytest
```

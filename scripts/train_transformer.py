"""Train the tiny Transformer. Default config: configs/tiny_transformer_cpu.yaml.

    python scripts/train_transformer.py --local
    python scripts/train_transformer.py configs/cuda_cuda_mps.yaml
"""

import sys

from meshtrain.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    config = args.pop(0) if args and args[0].endswith(".yaml") else "configs/tiny_transformer_cpu.yaml"
    sys.exit(main(["train", config, *args]))

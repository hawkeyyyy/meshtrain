"""Train the test MLP. Locally (all stages as processes):  python scripts/train_mlp.py --local
On a cluster (coordinator from $MESHTRAIN_COORDINATOR):    python scripts/train_mlp.py"""

import sys

from meshtrain.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", "configs/local_cpu.yaml", *sys.argv[1:]]))

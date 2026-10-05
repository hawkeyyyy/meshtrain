"""Thin wrapper: equivalent to `meshtrain coordinator start ...`."""

import sys

from meshtrain.cli import main

if __name__ == "__main__":
    sys.exit(main(["coordinator", "start", *sys.argv[1:]]))

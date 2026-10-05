"""Thin wrapper: equivalent to `meshtrain cluster benchmark ...`."""

import sys

from meshtrain.cli import main

if __name__ == "__main__":
    sys.exit(main(["cluster", "benchmark", *sys.argv[1:]]))

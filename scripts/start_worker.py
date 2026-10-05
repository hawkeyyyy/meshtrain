"""Thin wrapper: equivalent to `meshtrain worker join ...`."""

import sys

from meshtrain.cli import main

if __name__ == "__main__":
    sys.exit(main(["worker", "join", *sys.argv[1:]]))

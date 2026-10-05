import os

import pytest
import torch

os.environ.setdefault("MESHTRAIN_QUIET", "1")

HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "cuda" in item.keywords and not HAS_CUDA:
            item.add_marker(pytest.mark.skip(reason="no physical CUDA device available"))
        if "mps" in item.keywords and not HAS_MPS:
            item.add_marker(pytest.mark.skip(reason="no Apple MPS device available"))

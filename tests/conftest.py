import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure() -> None:
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _seed_random_generators() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


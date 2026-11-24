"""Shared pytest configuration and fixtures.

The ``device``, ``dtype``, ``seed`` and ``generator`` fixtures are shared by
every test module; a module that needs another value (``test_model.py`` runs
in float32, ``test_regressions.py`` uses its own seed) overrides the fixture
locally.
"""

import sys
from pathlib import Path

import pytest
import torch


# make the helpers of tests/utils.py importable as ``utils``
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    return torch.float64


@pytest.fixture(scope="session")
def seed():
    return 777


@pytest.fixture
def generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

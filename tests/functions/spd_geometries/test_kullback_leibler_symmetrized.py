import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_linalg as spd_linalg

import yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized as kullback_leibler_symmetrized

from yetanotherspdnet.random.spd import random_SPD, random_DPD
from yetanotherspdnet.random.stiefel import random_stiefel

from utils import is_symmetric, is_spd


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="function")
def generator(device):
    generator = torch.Generator(device=device)
    generator.manual_seed(777)
    return generator

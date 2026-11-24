"""Deprecated aliases: they still work, match their replacement and warn once."""

import warnings

import pytest
import torch

from yetanotherspdnet import _deprecation
from yetanotherspdnet.functions import spd_linalg
from yetanotherspdnet.random.spd import random_SPD


@pytest.fixture(autouse=True)
def _reset_warned():
    """Each test sees the first use of every alias."""
    _deprecation._WARNED.clear()
    yield
    _deprecation._WARNED.clear()


@pytest.fixture
def spd_batch(dtype, generator):
    return random_SPD(4, 5, cond=10, dtype=dtype, generator=generator)


def test_alias_warns_once(spd_batch):
    with pytest.warns(DeprecationWarning, match="invm_SPD is deprecated"):
        spd_linalg.invm_SPD(spd_batch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spd_linalg.invm_SPD(spd_batch)


def test_alias_is_marked_deprecated():
    assert hasattr(spd_linalg.invm_SPD, "__deprecated__")
    assert hasattr(spd_linalg.InvmSPD, "__deprecated__")


def test_invm_SPD(spd_batch):
    with pytest.warns(DeprecationWarning):
        inv, eigvals, eigvecs = spd_linalg.invm_SPD(spd_batch)
    torch.testing.assert_close(inv, spd_linalg.inv_SPD(spd_batch))
    torch.testing.assert_close(eigvals, torch.linalg.eigvalsh(spd_batch))
    assert eigvecs.shape == spd_batch.shape


def test_InvmSPD(spd_batch):
    data = spd_batch.clone().requires_grad_(True)
    with pytest.warns(DeprecationWarning, match="InvmSPD is deprecated"):
        out = spd_linalg.InvmSPD.apply(data)
    torch.testing.assert_close(out, spd_linalg.inv_SPD(spd_batch))
    out.sum().backward()
    data_ref = spd_batch.clone().requires_grad_(True)
    spd_linalg.inv_SPD(data_ref).sum().backward()
    torch.testing.assert_close(data.grad, data_ref.grad)


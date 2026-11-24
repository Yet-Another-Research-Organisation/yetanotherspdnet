"""Deprecated aliases: they still work, match their replacement and warn once."""

import warnings

import pytest
import torch

from yetanotherspdnet import _deprecation
from yetanotherspdnet.functions import spd_linalg
from yetanotherspdnet.functions.spd_geometries import (
    affine_invariant,
    kullback_leibler_symmetrized,
)
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


# ---------------------------------------------------------------------------
# adaptive GAH geodesic and mean of two points: the affine-invariant geodesic
# ---------------------------------------------------------------------------
def _geodesic_pair(spd_batch):
    point1 = spd_batch[0].clone().requires_grad_(True)
    point2 = spd_batch[1].clone().requires_grad_(True)
    return point1, point2


@pytest.mark.parametrize(
    "alias_name, canonical_name, manual",
    [
        (
            "adaptive_geometric_arithmetic_harmonic_geodesic",
            "affine_invariant_geodesic",
            False,
        ),
        (
            "AdaptiveGeometricArithmeticHarmonicGeodesic",
            "AffineInvariantGeodesic",
            True,
        ),
    ],
)
def test_adaptive_gah_geodesic_aliases(
    alias_name, canonical_name, manual, spd_batch, dtype
):
    alias = getattr(kullback_leibler_symmetrized, alias_name)
    canonical = getattr(affine_invariant, canonical_name)
    if manual:
        alias, canonical = alias.apply, canonical.apply
    results = []
    for fun in (alias, canonical):
        point1, point2 = _geodesic_pair(spd_batch)
        t = torch.tensor(0.3, dtype=dtype, requires_grad=True)
        out = fun(point1, point2, t)
        out.sum().backward()
        results.append((out, point1.grad, point2.grad, t.grad))
    for got, expected in zip(*results):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_adaptive_gah_geodesic_alias_warns(spd_batch):
    with pytest.warns(DeprecationWarning, match="AffineInvariantGeodesic"):
        kullback_leibler_symmetrized.AdaptiveGeometricArithmeticHarmonicGeodesic.apply(
            spd_batch[0], spd_batch[1], 0.5
        )
    with pytest.warns(DeprecationWarning, match="affine_invariant_geodesic"):
        kullback_leibler_symmetrized.adaptive_geometric_arithmetic_harmonic_geodesic(
            spd_batch[0], spd_batch[1], 0.5
        )


@pytest.mark.parametrize("manual", [False, True])
def test_affine_invariant_mean_2points_aliases(manual, spd_batch):
    if manual:
        alias = affine_invariant.AffineInvariantMean2Points.apply
        canonical = affine_invariant.AffineInvariantGeodesic.apply
    else:
        alias = affine_invariant.affine_invariant_mean_2points
        canonical = affine_invariant.affine_invariant_geodesic
    point1, point2 = _geodesic_pair(spd_batch)
    with pytest.warns(DeprecationWarning, match="mean_2points|Mean2Points"):
        out = alias(point1, point2)
    out.sum().backward()
    ref1, ref2 = _geodesic_pair(spd_batch)
    ref = canonical(ref1, ref2, 0.5)
    ref.sum().backward()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(point1.grad, ref1.grad, rtol=0, atol=0)
    torch.testing.assert_close(point2.grad, ref2.grad, rtol=0, atol=0)
    # the geometric mean of two points is symmetric in its arguments
    torch.testing.assert_close(out, canonical(spd_batch[1], spd_batch[0], 0.5))

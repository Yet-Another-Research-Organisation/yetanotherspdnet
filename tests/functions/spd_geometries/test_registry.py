"""Per-geometry registry used by the batch normalizations."""

import pytest
import torch
from torch.testing import assert_close

from yetanotherspdnet.functions.spd_geometries.registry import (
    GEOMETRIES,
    DualPath,
    get_geometry,
)
from yetanotherspdnet.nn import BatchNormSPDMean, BatchNormSPDMeanScalarVariance
from yetanotherspdnet.random.spd import random_SPD


MEAN_TYPES = list(GEOMETRIES)


def test_registered_mean_types():
    assert MEAN_TYPES == [
        "affine_invariant",
        "affine_invariant_gevd",
        "log_euclidean",
        "arithmetic",
        "harmonic",
        "geometric_arithmetic_harmonic",
        "adaptive_geometric_arithmetic_harmonic",
        "bures_wasserstein",
    ]


def test_unknown_mean_type():
    with pytest.raises(ValueError, match="mean_type must be one of"):
        get_geometry("euclidean")
    with pytest.raises(ValueError, match="mean_type must be one of"):
        BatchNormSPDMean(3, mean_type="euclidean")


def test_dual_path_select():
    path = DualPath(autograd=abs, manual=round)
    assert path.select(True) is abs
    assert path.select(False) is round
    unsupported = DualPath(autograd=None, manual=round, unsupported_reason="no")
    with pytest.raises(ValueError, match="no"):
        unsupported.select(True)


@pytest.mark.parametrize("mean_type", MEAN_TYPES)
def test_manual_and_autograd_paths_agree(mean_type, dtype, generator):
    """Each operation gives the same value through both paths."""
    geometry = get_geometry(mean_type)
    data = random_SPD(4, 6, cond=10, dtype=dtype, generator=generator)
    mean_args = (torch.tensor(0.3, dtype=dtype),) if geometry.learnable_t else ()
    means = [geometry.mean.select(ag)(data, *mean_args) for ag in (True, False)]
    assert_close(means[0], means[1])
    stds = [geometry.std.select(ag)(data, means[0]) for ag in (True, False)]
    assert_close(stds[0], stds[1])
    geodesics = [
        geometry.geodesic.select(ag)(data[0], data[1], 0.3) for ag in (True, False)
    ]
    assert_close(geodesics[0], geodesics[1])
    if geometry.transport is not None:
        transported = geometry.transport.manual(data, means[0], data[2])
        assert transported.shape == data.shape


def test_mean_options():
    assert get_geometry("affine_invariant").mean_kwargs(None) == {}
    assert get_geometry("affine_invariant").mean_kwargs({"n_iterations": 3}) == {
        "n_iterations": 3
    }
    # the BW mean defaults to one fixed-point iteration
    assert get_geometry("bures_wasserstein").mean_kwargs(None) == {"n_iterations": 1}
    # options of other means are ignored
    assert get_geometry("log_euclidean").mean_kwargs({"n_iterations": 3}) == {}


@pytest.mark.parametrize("use_autograd", [False, True])
def test_unsupported_transport_fails_at_construction(use_autograd):
    with pytest.raises(ValueError, match="not available"):
        BatchNormSPDMean(
            3,
            mean_type="bures_wasserstein",
            norm_and_bias_mode="parallel_transport",
            use_autograd=use_autograd,
        )


@pytest.mark.parametrize("mean_type", ["affine_invariant", "log_euclidean"])
def test_transport_autograd_unsupported_fails_at_construction(mean_type):
    with pytest.raises(ValueError, match="NaN gradients"):
        BatchNormSPDMean(
            3,
            mean_type=mean_type,
            norm_and_bias_mode="parallel_transport",
            use_autograd=True,
        )


@pytest.mark.parametrize("use_autograd", [False, True])
@pytest.mark.parametrize("mean_type", MEAN_TYPES)
def test_batchnorm_uses_registry(mean_type, use_autograd):
    """The layers take their operations from the registry."""
    geometry = get_geometry(mean_type)
    bn = BatchNormSPDMeanScalarVariance(
        3, mean_type=mean_type, norm_strategy="minibatch", use_autograd=use_autograd
    )
    assert bn.geometry is geometry
    assert bn.std_fun is geometry.std.select(use_autograd)
    assert bn.regularize_mean_fun is geometry.geodesic.select(use_autograd)
    # the running mean is updated without gradient
    assert bn.adaptive_mean_fun is geometry.geodesic.autograd

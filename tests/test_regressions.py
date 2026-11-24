"""Regression tests for bugs fixed after the code audit of November 2026.

Each test failed before its fix.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_close

from utils import is_spd
from yetanotherspdnet.functions import spd_linalg
from yetanotherspdnet.functions.spd_geometries import (
    affine_invariant,
    bures_wasserstein,
    euclidean,
    kullback_leibler,
    kullback_leibler_symmetrized,
    log_euclidean,
)
from yetanotherspdnet.model import GBWBNRResNet, RResNet, SPDnet
from yetanotherspdnet.nn.base import BiMap
from yetanotherspdnet.nn.batchnorm import BatchNormSPDMeanScalarVariance
from yetanotherspdnet.nn.rresnet_layers import SpectralVectorField
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.random.stiefel import random_stiefel


DTYPE = torch.float64


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(2026)


# ---------------------------------------------------------------------------
# Whitening.backward printed its gradients
# ---------------------------------------------------------------------------
def test_whitening_backward_prints_nothing(capsys, generator):
    X = random_SPD(5, 4, cond=10, dtype=DTYPE, generator=generator)
    G = random_SPD(5, 1, cond=10, dtype=DTYPE, generator=generator)
    X.requires_grad_(True)
    G.requires_grad_(True)
    spd_linalg.Whitening.apply(X, G).sum().backward()
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# BatchNormSPDMeanScalarVariance had no std for the GEVD and adaptive GAH means
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mean_type, std_reference",
    [
        ("affine_invariant_gevd", affine_invariant.affine_invariant_std_scalar),
        (
            "adaptive_geometric_arithmetic_harmonic",
            kullback_leibler_symmetrized.symmetrized_kullback_leibler_std_scalar,
        ),
    ],
)
@pytest.mark.parametrize("use_autograd", [False, True])
def test_scalar_variance_std_for_all_mean_types(
    mean_type, std_reference, use_autograd, generator
):
    layer = BatchNormSPDMeanScalarVariance(
        6, mean_type=mean_type, use_autograd=use_autograd, dtype=DTYPE
    )
    X = random_SPD(6, 10, cond=10, dtype=DTYPE, generator=generator)
    X.requires_grad_(True)
    output = layer(X)
    output.sum().backward()
    assert is_spd(output.detach())
    assert torch.isfinite(X.grad).all()
    mean = layer.mean_fun(X.detach())
    assert_close(layer.std_fun(X.detach(), mean), std_reference(X.detach(), mean))


# ---------------------------------------------------------------------------
# the minibatch std ignored the momentum schedule (minibatch_mode)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("minibatch_mode", ["decay", "growth"])
def test_minibatch_std_uses_momentum_schedule(minibatch_mode, generator):
    layer = BatchNormSPDMeanScalarVariance(
        5,
        mean_type="log_euclidean",
        norm_strategy="minibatch",
        minibatch_mode=minibatch_mode,
        minibatch_momentum=0.1,
        minibatch_maxstep=10,
        dtype=DTYPE,
    )
    for _ in range(3):
        X = random_SPD(5, 8, cond=10, dtype=DTYPE, generator=generator)
        momentum = layer.get_minibatch_momentum()
        std_regularizer = layer.std_regularizer.clone()
        _, (_, std_batch) = layer._get_norm_parameters(X)
        expected = torch.sqrt(
            (1 - momentum) * std_regularizer**2 + momentum * std_batch**2
        )
        assert_close(layer.std_regularizer, expected.detach())
        layer.training_step += 1


# ---------------------------------------------------------------------------
# dynamic RResNet hook: wrong signature, and Q jumped after a reference update
# ---------------------------------------------------------------------------
def _check_dynamic_update(module, name, make_input, n_steps):
    """Train with a real optimizer; check the tensor across reference updates."""
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    after_step = {}

    # registered before the module hook, so it runs between step and recentring
    def record(_optimizer, *_args, **_kwargs):
        with torch.no_grad():
            after_step["value"] = getattr(module, name).detach().clone()

    optimizer.register_step_post_hook(record)
    module.register_optimizer_hook(optimizer)
    parametrizations = getattr(module.parametrizations, name)
    for step in range(1, 5):
        optimizer.zero_grad()
        output = module(make_input())
        before_step = getattr(module, name).detach().clone()
        # a loss that depends on the orientation (||Q diag(f) Q^T||^2 does not)
        weights = torch.arange(output[0].numel(), dtype=DTYPE).reshape(output[0].shape)
        (output * weights).sum().backward()
        optimizer.step()
        value = getattr(module, name).detach()
        # the optimizer moved the tensor
        assert not torch.allclose(after_step["value"], before_step)
        if step % n_steps == 0:
            # reference point = the tensor after the step, tangent vector reset:
            # the tensor is continuous across the update, and the step is kept
            assert module.current_ref_step == 0
            assert_close(parametrizations.original, torch.zeros_like(value))
            assert_close(parametrizations[0].reference_point, after_step["value"])
            assert_close(value, after_step["value"])
        else:
            assert module.current_ref_step == step % n_steps
            assert (parametrizations.original != 0).any()
            assert_close(value, after_step["value"])


def test_spectral_vector_field_dynamic_update_with_optimizer(generator):
    vf = SpectralVectorField(
        5,
        stiefel_parametrization_mode="dynamic",
        stiefel_n_steps_ref_update=2,
        dtype=DTYPE,
        generator=generator,
    )
    _check_dynamic_update(
        vf, "Q", lambda: random_SPD(5, 8, cond=10, dtype=DTYPE, generator=generator), 2
    )


def test_bimap_dynamic_update_with_optimizer(generator):
    bimap = BiMap(5, 3, parametrization_mode="dynamic", n_steps_ref_update=2)
    _check_dynamic_update(
        bimap,
        "weight",
        lambda: random_SPD(5, 8, cond=10, dtype=DTYPE, generator=generator),
        2,
    )


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_spectral_vector_field_initial_q_is_the_sampled_one(mode):
    # the Stiefel parametrization had a random reference point of its own, so
    # the initial Q was a retraction of the sampled one, not the sampled one
    vf = SpectralVectorField(
        5,
        stiefel_parametrization_mode=mode,
        dtype=DTYPE,
        generator=torch.Generator().manual_seed(3),
    )
    sampled = random_stiefel(
        5, 5, n_matrices=1, dtype=DTYPE, generator=torch.Generator().manual_seed(3)
    ).squeeze(0)
    assert_close(vf.Q, sampled, rtol=0, atol=1e-14)
    assert_close(vf.parametrizations.Q.original, torch.zeros_like(sampled))


# ---------------------------------------------------------------------------
# ArithmeticMean.backward built its gradient in float32
# ---------------------------------------------------------------------------
def test_arithmetic_mean_backward_keeps_dtype(generator):
    X = random_SPD(3, 7, cond=10, dtype=DTYPE, generator=generator)
    X.requires_grad_(True)
    grad_output = torch.full((3, 3), 1 / 3, dtype=DTYPE)
    euclidean.ArithmeticMean.apply(X).backward(grad_output)
    assert_close(X.grad, (grad_output / 7).expand(7, 3, 3), rtol=1e-15, atol=0)


# ---------------------------------------------------------------------------
# PowmSPD failed with a python exponent (save_for_backward of a float)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exponent", [0.5, -1.5, 2])
def test_powm_python_exponent_backward(exponent, generator):
    X = random_SPD(4, 3, cond=10, dtype=DTYPE, generator=generator)
    X_manual = X.clone().requires_grad_(True)
    X_auto = X.clone().requires_grad_(True)
    spd_linalg.PowmSPD.apply(X_manual, exponent).sum().backward()
    spd_linalg.powm_SPD(X_auto, exponent)[0].sum().backward()
    assert_close(X_manual.grad, X_auto.grad)


def test_powm_tensor_exponent_gradcheck(generator):
    X = random_SPD(4, 3, cond=10, dtype=DTYPE, generator=generator)
    X.requires_grad_(True)
    exponent = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    # symmetric perturbations only (the input is an SPD matrix)
    assert gradcheck(
        lambda x, p: spd_linalg.PowmSPD.apply((x + x.mT) / 2, p), (X, exponent)
    )


# ---------------------------------------------------------------------------
# model hash: unstable (generator in SPDnet repr) or colliding (RResNet,
# GBWBNRResNet repr without most hyper-parameters)
# ---------------------------------------------------------------------------
def _hash(cls, seed=0, **kwargs):
    return cls(**kwargs, generator=torch.Generator().manual_seed(seed)).get_model_hash()


@pytest.mark.parametrize(
    "cls, config, other",
    [
        (
            SPDnet,
            {"input_dim": 6, "hidden_layers_size": [4], "output_dim": 2},
            {"batchnorm_bw_options": {"bw_theta": 1.0}},
        ),
        (
            RResNet,
            {
                "input_dim": 6,
                "hidden_layers_size": [4],
                "n_residual_blocks": [1],
                "output_dim": 2,
            },
            {"spectrum_hidden_dim": 5},
        ),
        (
            RResNet,
            {
                "input_dim": 6,
                "hidden_layers_size": [4],
                "n_residual_blocks": [1],
                "output_dim": 2,
            },
            {"residual_metric": "log_euclidean"},
        ),
        (
            GBWBNRResNet,
            {"input_dim": 6, "hidden_dim": 4, "output_dim": 2},
            {"batchnorm_bw_options": {"bw_theta": 1.0}},
        ),
        (
            GBWBNRResNet,
            {"input_dim": 6, "hidden_dim": 4, "output_dim": 2},
            {"batchnorm_momentum": 0.2},
        ),
    ],
)
def test_model_hash_stable_and_config_dependent(cls, config, other):
    assert _hash(cls, 0, **config) == _hash(cls, 1, **config)
    assert _hash(cls, 0, **config) != _hash(cls, 0, **config, **other)


# ---------------------------------------------------------------------------
# docs: dualpath pairing of CamelCase functions, misspelled transport name
# ---------------------------------------------------------------------------
def _load_dualpath_pairing():
    path = Path(__file__).parents[1] / "docs" / "_ext" / "dualpath_pairing.py"
    spec = importlib.util.spec_from_file_location("dualpath_pairing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "module, pair",
    [
        (affine_invariant, ("affine_invariant_mean", "AffineInvariantMean")),
        (
            affine_invariant,
            ("affine_invariant_parallel_transport", "AffineInvariantParallelTransport"),
        ),
        (log_euclidean, ("log_euclidean_mean", "LogEuclideanMean")),
        (
            log_euclidean,
            (
                "log_euclidean_parallel_transport_SPD",
                "LogEuclideanParallelTransportSPD",
            ),
        ),
        (spd_linalg, ("sqrtm_SPD", "SqrtmSPD")),
    ],
)
def test_dualpath_pairs_camelcase_functions(module, pair):
    pairing = _load_dualpath_pairing()
    rows = pairing.pair_members(pairing.public_members(module))
    assert pair in rows
    names = {name for row in rows for name in row}
    assert "LogEuclideanParalleTransportSPD" not in names


def test_log_euclidean_transport_deprecated_alias(generator):
    X = random_SPD(4, 3, cond=10, dtype=DTYPE, generator=generator)
    G1, G2 = random_SPD(4, 2, cond=10, dtype=DTYPE, generator=generator)
    with pytest.warns(DeprecationWarning, match="LogEuclideanParallelTransportSPD"):
        old = log_euclidean.LogEuclideanParalleTransportSPD(X, G1, G2)
    assert_close(old, log_euclidean.LogEuclideanParallelTransportSPD(X, G1, G2))


# ---------------------------------------------------------------------------
# std = sqrt(variance) without clamp, and std backwards dividing by std = 0
# ---------------------------------------------------------------------------
STD_FUNCTIONS = [
    affine_invariant.AffineInvariantStdScalar.apply,
    log_euclidean.LogEuclideanStdScalar.apply,
    bures_wasserstein.BuresWassersteinStdScalar.apply,
    kullback_leibler.LeftKullbackLeiblerStdScalar.apply,
    kullback_leibler.RightKullbackLeiblerStdScalar.apply,
    kullback_leibler_symmetrized.SymmetrizedKullbackLeiblerStdScalar.apply,
]


@pytest.mark.parametrize("std_fun", STD_FUNCTIONS)
@pytest.mark.parametrize("point", ["identity", "random"])
def test_std_of_identical_matrices_is_finite(std_fun, point, generator):
    if point == "identity":
        M = torch.eye(5, dtype=DTYPE)
    else:
        M = random_SPD(5, 1, cond=100, dtype=DTYPE, generator=generator)
    X = M.expand(6, 5, 5).clone().requires_grad_(True)
    G = M.clone().requires_grad_(True)
    std = std_fun(X, G)
    std.backward()
    assert torch.isfinite(std)
    assert std >= 0
    assert torch.isfinite(X.grad).all()
    assert torch.isfinite(G.grad).all()


@pytest.mark.parametrize(
    "std_fun",
    [
        affine_invariant.affine_invariant_std_scalar,
        log_euclidean.log_euclidean_std_scalar,
        bures_wasserstein.bures_wasserstein_std_scalar,
    ],
)
def test_std_forward_of_identical_matrices_is_finite(std_fun, generator):
    M = random_SPD(5, 1, cond=100, dtype=DTYPE, generator=generator)
    std = std_fun(M.expand(6, 5, 5), M)
    assert torch.isfinite(std)
    assert std >= 0


# ---------------------------------------------------------------------------
# GBWBN created an unused Covbias parameter
# ---------------------------------------------------------------------------
def test_bures_wasserstein_batchnorm_has_no_covbias():
    layer = BatchNormSPDMeanScalarVariance(
        4, mean_type="bures_wasserstein", dtype=DTYPE
    )
    names = {name for name, _ in layer.named_parameters()}
    assert not any("Covbias" in name for name in names)
    assert len(names) == 3  # stdScalarbias, bw_M, bw_G


@pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
def test_bures_wasserstein_batchnorm_loads_old_checkpoint(parametrization_mode):
    layer = BatchNormSPDMeanScalarVariance(
        4,
        mean_type="bures_wasserstein",
        parametrization_mode=parametrization_mode,
        dtype=DTYPE,
    )
    # checkpoint saved by a version that created Covbias in this mode
    state_dict = layer.state_dict()
    state_dict["parametrizations.Covbias.original"] = torch.eye(4, dtype=DTYPE)
    if parametrization_mode == "dynamic":
        state_dict["parametrizations.Covbias.0.reference_point"] = torch.eye(
            4, dtype=DTYPE
        )
    layer.load_state_dict(state_dict)


def test_bures_wasserstein_batchnorm_dynamic_optimizer_step(generator):
    layer = BatchNormSPDMeanScalarVariance(
        4,
        mean_type="bures_wasserstein",
        parametrization_mode="dynamic",
        n_steps_ref_update=1,
        dtype=DTYPE,
    )
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    layer.register_optimizer_hook(optimizer)
    for _ in range(2):
        X = random_SPD(4, 8, cond=10, dtype=DTYPE, generator=generator)
        optimizer.zero_grad()
        layer(X).sum().backward()
        optimizer.step()

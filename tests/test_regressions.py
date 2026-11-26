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


# ---------------------------------------------------------------------------
# GBWBN: the bias exponential Exp_G left its domain (I + Z > 0) and produced a
# numerically singular output, which LogEig turned into NaN (E1 benchmark,
# HDM05 and GSOFF): SPDnet now places GBWBN before the ReEig, as the reference
# code (the projection of Exp_G stays an opt-in of the geometry functions)
# ---------------------------------------------------------------------------
def _diagonal_batch(rows):
    return torch.diag_embed(torch.tensor(rows, dtype=DTYPE))


def _gbwbn_at_the_boundary(use_autograd):
    """GBWBN layer (theta = 1, M = I) whose bias transport map is singular.

    Diagonal (commuting) data: the transport map of Exp_G is diagonal,
    T = 1 + (|1 + f (c^{1/2} - 1)| - 1) / g^{1/2} with c the centered data and
    f = s / sqrt(var + eps) the scaling factor; s is chosen so that T = 0 for
    the first coordinate of the first matrix.
    """
    X = _diagonal_batch(
        [[0.3, 1.0, 2.0], [1.5, 0.8, 1.0], [1.2, 1.3, 0.6], [0.9, 1.1, 1.4]]
    )
    g = 0.01
    layer = BatchNormSPDMeanScalarVariance(
        3,
        mean_type="bures_wasserstein",
        bw_theta=1.0,
        use_autograd=use_autograd,
        dtype=DTYPE,
    )
    layer._move_reference_point("bw_G", torch.diag(torch.tensor([g, 1.0, 1.0])))
    bw = bures_wasserstein
    mean = bw.bures_wasserstein_mean(X)
    std = bw.bures_wasserstein_std_scalar(X, mean)
    c = bw.bures_wasserstein_center(X, mean)[0, 0, 0]
    assert c < 1
    factor = g**0.5 / (1 - c**0.5)
    with torch.no_grad():
        layer.stdScalarbias = factor * torch.sqrt(std**2 + 1e-5)
    return layer, X, g


@pytest.mark.parametrize("use_autograd", [False, True])
def test_gbwbn_singular_output_is_floored_by_reeig(use_autograd):
    """GBWBN at the boundary of its bias exponential: singular output, which a
    LogEig cannot take; the ReEig placed after it (BiMap -> GBWBN -> ReEig, the
    SPDnet default for GBWBN) floors it, and LogEig and the gradients are finite."""
    from yetanotherspdnet.nn import LogEig, ReEig

    layer, X, _ = _gbwbn_at_the_boundary(use_autograd)
    output = layer(X)
    eigvals = torch.linalg.eigvalsh(output.detach())
    # the paper's formula: singular (the formula folds through 0)
    assert (eigvals[:, 0] / eigvals[:, -1]).min() < 1e-10
    features = LogEig(use_autograd=use_autograd)(
        ReEig(eps=1e-4, use_autograd=use_autograd)(output)
    )
    assert torch.isfinite(features).all()
    if use_autograd:
        # autograd through eigh is undefined at M = I (repeated eigenvalues)
        return
    features.square().sum().backward()
    for name, parameter in layer.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name


def test_spdnet_places_gbwbn_before_reeig():
    from yetanotherspdnet.nn import ReEig

    gbwbn = {
        "batchnorm": True,
        "batchnorm_type": "mean_var_scalar",
        "batchnorm_mean_type": "bures_wasserstein",
    }
    kinds = lambda model: [type(m).__name__ for m in model.spdnet_layers]
    model = SPDnet(8, [6, 4], 3, **gbwbn, dtype=DTYPE)
    layers = list(model.spdnet_layers)
    for i, layer in enumerate(layers):
        if isinstance(layer, BatchNormSPDMeanScalarVariance):
            assert isinstance(layers[i + 1], ReEig)
    assert kinds(model)[-2:] == ["ReEig", "LogEig"]
    # explicit placement of the other batch norms, and of GBWBN after the ReEig
    other = SPDnet(8, [6, 4], 3, batchnorm=True, dtype=DTYPE)
    assert kinds(other)[1:3] == ["ReEig", "ParametrizedBatchNormSPDMean"]
    after = SPDnet(8, [6, 4], 3, **gbwbn, batchnorm_position="after_reeig", dtype=DTYPE)
    assert kinds(after)[-2:] == ["ParametrizedBatchNormSPDMeanScalarVariance", "LogEig"]
    with pytest.raises(ValueError, match="batchnorm_position"):
        SPDnet(8, [6], 3, batchnorm=True, batchnorm_position="before", dtype=DTYPE)


def test_bw_bias_projection_is_exact_inside_the_domain(generator):
    data = random_SPD(5, 6, cond=4, dtype=DTYPE, generator=generator)
    bias = random_SPD(5, 1, cond=4, dtype=DTYPE, generator=generator)
    bw = bures_wasserstein
    for fun in (bw.bures_wasserstein_bias, bw.BuresWassersteinBias):
        # all the transport maps are above the bound: unchanged, bit for bit
        assert torch.equal(fun(data, bias, 1e-3), fun(data, bias))


def test_bw_bias_projection_gradients(generator):
    """Manual backward of the projected bias = autograd = finite differences."""
    bw = bures_wasserstein
    data = random_SPD(4, 3, cond=50, dtype=DTYPE, generator=generator)
    eigvals = torch.tensor([0.02, 0.3, 1.0, 1.5], dtype=DTYPE)
    Q = torch.linalg.qr(torch.randn(4, 4, dtype=DTYPE, generator=generator))[0]
    bias = Q @ torch.diag(eigvals) @ Q.mT
    tau = 0.2
    # the projection acts on some transport maps (none is at the bound)
    Z = bw.LyapunovSolveSPD.apply(
        bias,
        bw.bures_wasserstein_parallel_transport_from_identity(
            bw.bures_wasserstein_log_identity(data), bias
        ),
    )
    t_min = torch.linalg.eigvalsh(torch.eye(4, dtype=DTYPE) + Z)[:, 0]
    assert (t_min < tau - 1e-2).any()
    assert ((t_min - tau).abs() > 1e-3).all()

    data.requires_grad_(True)
    bias.requires_grad_(True)
    grads = []
    for fun in (bw.BuresWassersteinBias, bw.bures_wasserstein_bias):
        out = fun(data, bias, tau)
        assert torch.linalg.eigvalsh(out).min() >= 0.99 * tau**2 * eigvals[0]
        grads.append(torch.autograd.grad(out.square().sum(), (data, bias)))
    for manual, auto in zip(*grads):
        assert_close(manual, auto)
    sym = lambda A: (A + A.mT) / 2
    assert gradcheck(
        lambda x, b: bw.BuresWassersteinBias(sym(x), sym(b), tau),
        (data, bias),
    )


def test_gbwbn_short_training_stays_finite():
    """Deterministic SPDnet + GBWBN training on synthetic data: NaN loss at epoch 23
    with the GBWBN after the ReEig (the placement before the fix, still available as
    batchnorm_position="after_reeig"), finite with the default GBWBN -> ReEig."""
    seed, n = 2, 20
    g = torch.Generator().manual_seed(seed)
    batches, labels = [], []
    for label in range(3):
        C = random_SPD(n, 1, cond=50, dtype=DTYPE, generator=g)
        L = torch.linalg.cholesky(C)
        A = torch.randn(40, n, 3 * n, generator=g, dtype=DTYPE)
        batches.append(L @ (A @ A.mT / (3 * n)) @ L.mT)
        labels.append(torch.full((40,), label))
    X, y = torch.cat(batches), torch.cat(labels)
    torch.manual_seed(seed)
    model = SPDnet(
        n,
        [16, 12],
        3,
        batchnorm=True,
        batchnorm_type="mean_var_scalar",
        batchnorm_mean_type="bures_wasserstein",
        batchnorm_momentum=0.1,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    shuffle = torch.Generator().manual_seed(seed)
    for epoch in range(25):
        perm = torch.randperm(len(y), generator=shuffle)
        for i in range(0, len(y), 16):
            idx = perm[i : i + 16]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# GBWBN computes in float64 for float32 data (its domain projection would
# otherwise need the float32 bound, tau ~ 0.14 at theta = 0.5)
# ---------------------------------------------------------------------------
def _gbwbn_pair(norm_strategy="classical"):
    """The same GBWBN layer in float64 and in float32 (parameters off the identity)."""
    layers = {}
    for dtype in (torch.float64, torch.float32):
        layers[dtype] = BatchNormSPDMeanScalarVariance(
            5,
            mean_type="bures_wasserstein",
            norm_strategy=norm_strategy,
            minibatch_mode="constant" if norm_strategy == "minibatch" else "",
            dtype=dtype,
        )
    gen = torch.Generator().manual_seed(7)
    M = random_SPD(5, 1, cond=3, dtype=torch.float32, generator=gen)
    G = random_SPD(5, 1, cond=3, dtype=torch.float32, generator=gen)
    for dtype, layer in layers.items():
        layer._move_reference_point("bw_M", M.to(dtype))
        layer._move_reference_point("bw_G", G.to(dtype))
        with torch.no_grad():
            layer.stdScalarbias = torch.tensor(1.3, dtype=dtype)
    X = random_SPD(5, 16, cond=10, dtype=torch.float32, generator=gen)
    return layers[torch.float64], layers[torch.float32], X


@pytest.mark.parametrize("norm_strategy", ["classical", "minibatch"])
def test_gbwbn_float32_data_matches_float64(norm_strategy):
    layer64, layer32, X = _gbwbn_pair(norm_strategy)

    def relative_error(out32, out64):
        # float32 rounding of the input, the output and the float32 buffers
        # (running statistics, minibatch regularizers) between steps
        out32, out64 = out32.detach().double(), out64.detach()
        diff = torch.linalg.matrix_norm(out32 - out64)
        return float((diff / torch.linalg.matrix_norm(out64)).max())

    for _ in range(3):  # training steps (running statistics), then evaluation
        out32, out64 = layer32(X), layer64(X.double())
        assert out32.dtype == torch.float32
        assert relative_error(out32, out64) < 2e-6
    layer32.eval()
    layer64.eval()
    assert relative_error(layer32(X), layer64(X.double())) < 2e-6
    # parameters and buffers keep their dtype
    for name, value in layer32.state_dict().items():
        if value.is_floating_point():
            assert value.dtype == torch.float32, name


def test_gbwbn_float32_gradients_reach_parameters():
    layer64, layer32, X = _gbwbn_pair()
    X32 = X.clone().requires_grad_(True)
    X64 = X.double().requires_grad_(True)
    layer32(X32).square().sum().backward()
    layer64(X64).square().sum().backward()
    params64 = dict(layer64.named_parameters())
    for name, parameter in layer32.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == torch.float32, name
        assert_close(parameter.grad.double(), params64[name].grad, rtol=1e-4, atol=1e-5)
    assert X32.grad.dtype == torch.float32
    assert_close(X32.grad.double(), X64.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("bw_theta", [1.0, 0.5])
def test_gbwbn_float32_bias_step_runs_in_float64(monkeypatch, bw_theta):
    from yetanotherspdnet.functions.dispatch import DualPath
    from yetanotherspdnet.nn import batchnorm

    calls = []
    original = batchnorm.BURES_WASSERSTEIN_STEPS["bias"]

    def spy(data, bias_point, *args):
        calls.append((data.dtype, bias_point.dtype, args))
        return original.manual(data, bias_point, *args)

    monkeypatch.setitem(
        batchnorm.BURES_WASSERSTEIN_STEPS, "bias", DualPath(original.autograd, spy)
    )
    layer = BatchNormSPDMeanScalarVariance(
        4, mean_type="bures_wasserstein", bw_theta=bw_theta, dtype=torch.float32
    )
    X = random_SPD(4, 8, cond=10, dtype=torch.float32)
    assert layer(X).dtype == torch.float32
    # float64 tensors, and no projection of the bias exponential (paper formula)
    assert calls == [(torch.float64, torch.float64, ())]

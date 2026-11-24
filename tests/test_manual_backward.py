"""Manual backward is the norm: autograd-only list, gradchecks of the manual Functions.

With ``use_autograd=False`` (the default), no matrix decomposition may be
differentiated by autograd, except the operations listed in
``yetanotherspdnet.functions.dispatch.AUTOGRAD_ONLY``. The check walks the
autograd graph of every layer and model: operations inside a custom
``torch.autograd.Function.forward`` are not recorded, so any native
decomposition node found there is autograd through a decomposition.
"""

import importlib
import warnings

import pytest
import torch
from torch.autograd import gradcheck

from yetanotherspdnet.functions import dispatch
from yetanotherspdnet.functions.dispatch import (
    AUTOGRAD_ONLY,
    AutogradOnlyWarning,
    DualPath,
)
from yetanotherspdnet.functions.spd_geometries import bures_wasserstein as bw
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    AffineInvariantNorm,
    affine_invariant_norm,
)
from yetanotherspdnet.functions.spd_linalg import (
    EighClamp,
    EigvalshSymmetric,
    eigh_clamp,
    symmetrize,
)
from yetanotherspdnet.model import GBWBNRResNet, RResNet, SPDnet
from yetanotherspdnet.nn import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
    BiMap,
    LogEig,
    ReEig,
    ReEigBias,
    ResidualBlock,
    SpectralVectorField,
)


F64 = torch.float64
N = 4

# native autograd nodes of decompositions (and of torch's orthogonal
# parametrization), as named by torch
DECOMPOSITION_MARKERS = ("Linalg", "Cholesky", "Svd", "Qr", "Slogdet", "Eig")
# decomposition nodes coming from the AUTOGRAD_ONLY entries reachable by layers
ALLOWED_NODES = {
    # BiMap(parametrization_mode="static"): torch.nn.utils.parametrizations.orthogonal
    "LinalgHouseholderProductBackward0",
}


def decomposition_nodes(output: torch.Tensor) -> set[str]:
    """Names of the native decomposition nodes of the graph of *output*."""
    seen, found, stack = set(), set(), [output.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        if (
            name.endswith("Backward0")
            and "Norm" not in name
            and any(marker in name for marker in DECOMPOSITION_MARKERS)
        ):
            found.add(name)
        stack.extend(next_node for next_node, _ in node.next_functions)
    return found


def random_spd(n_matrices, n, kappa=10.0, seed=0, dtype=F64):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(n_matrices, n, n, dtype=dtype, generator=generator)
    )
    u = torch.rand(n_matrices, n, dtype=dtype, generator=generator)
    lam = kappa ** (u - 0.5)
    return (q * lam.unsqueeze(-2)) @ q.mT


MEAN_TYPES = [
    "affine_invariant",
    "affine_invariant_gevd",
    "log_euclidean",
    "arithmetic",
    "harmonic",
    "geometric_arithmetic_harmonic",
    "adaptive_geometric_arithmetic_harmonic",
    "bures_wasserstein",
]


def _configs():
    configs = []
    for cls in (BatchNormSPDMean, BatchNormSPDMeanScalarVariance):
        for mean_type in MEAN_TYPES:
            for strategy in ("classical", "minibatch"):
                kwargs = {"mean_type": mean_type, "norm_strategy": strategy}
                configs.append((cls, (N,), kwargs))
                if cls is BatchNormSPDMean and mean_type != "bures_wasserstein":
                    configs.append(
                        (
                            cls,
                            (N,),
                            {**kwargs, "norm_and_bias_mode": "parallel_transport"},
                        )
                    )
            configs.append(
                (cls, (N,), {"mean_type": mean_type, "parametrization_mode": "dynamic"})
            )
    configs += [
        (
            BatchNormSPDMeanScalarVariance,
            (N,),
            {"mean_type": "bures_wasserstein", "bw_theta": 1.0},
        ),
        (BiMap, (N, 3), {}),
        (BiMap, (N, 3), {"parametrization_mode": "dynamic"}),
        (ReEig, (), {}),
        (ReEigBias, (N,), {}),
        (LogEig, (), {}),
        (SpectralVectorField, (N,), {}),
        (ResidualBlock, (N,), {}),
        (ResidualBlock, (N,), {"metric": "log_euclidean"}),
        (SPDnet, (N, [3], 2), {"batchnorm": True}),
        (SPDnet, (N, [3], 2), {"batchnorm": True, "batchnorm_type": "mean_var_scalar"}),
        (RResNet, (N, [3], [2], 2), {"batchnorm": True}),
        (GBWBNRResNet, (N, 3, 2), {}),
    ]
    return configs


CONFIGS = _configs()


@pytest.mark.parametrize(
    ("cls", "args", "kwargs"),
    CONFIGS,
    ids=[f"{c.__name__}-{k}" for c, _, k in CONFIGS],
)
def test_no_autograd_through_decompositions(cls, args, kwargs):
    """Default (manual) graphs hold no decomposition node outside AUTOGRAD_ONLY."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutogradOnlyWarning)
        module = cls(*args, **kwargs)
    module.train()
    data = random_spd(6, N).requires_grad_(True)
    # two training steps: the second one uses the minibatch regularization
    found = decomposition_nodes(module(data)) | decomposition_nodes(module(data))
    assert found <= ALLOWED_NODES, found


def test_autograd_only_entries_resolve():
    """Each AUTOGRAD_ONLY entry names an existing object and gives a reason."""
    for name, reason in AUTOGRAD_ONLY.items():
        assert reason
        path = name.split("(")[0]
        module_name, _, attribute = path.rpartition(".")
        assert hasattr(importlib.import_module(module_name), attribute), name


def test_missing_manual_path_must_be_listed():
    """A DualPath without manual implementation must name an AUTOGRAD_ONLY entry."""
    with pytest.raises(ValueError, match="AUTOGRAD_ONLY"):
        DualPath(autograd=abs, manual=None)
    with pytest.raises(ValueError, match="AUTOGRAD_ONLY"):
        DualPath(autograd=abs, manual=None, autograd_only="not listed")


def test_autograd_only_warns_once(monkeypatch):
    """Requesting the manual path of an autograd-only operation warns once."""
    name = next(iter(AUTOGRAD_ONLY))
    monkeypatch.setattr(dispatch, "_WARNED", set())
    path = DualPath(autograd=abs, manual=None, autograd_only=name)
    with pytest.warns(AutogradOnlyWarning, match="no manual backward"):
        assert path.select(False) is abs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert path.select(False) is abs
    assert path.select(True) is abs


def test_bimap_static_warns(monkeypatch):
    """BiMap's static (torch) parametrization is the listed autograd-only path."""
    monkeypatch.setattr(dispatch, "_WARNED", set())
    with pytest.warns(AutogradOnlyWarning, match="BiMap"):
        BiMap(N, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BiMap(N, 3, parametrization_mode="dynamic")
        BiMap(N, 3, use_autograd=True)


# ---------------------------------------------------------------------------
# gradchecks of the manual Functions (float64)
# ---------------------------------------------------------------------------
def _spd(eigvals, seed=0):
    eigvals = torch.tensor(eigvals, dtype=F64)
    generator = torch.Generator().manual_seed(seed)
    n = eigvals.shape[-1]
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=F64, generator=generator))
    return (q * eigvals) @ q.T


SPECTRA = {
    "repeated": [2.0, 2.0, 2.0, 5.0],
    "nearly_repeated": [2.0, 2.0 + 1e-4, 2.0 + 2e-4, 5.0],
    "identity": [1.0, 1.0, 1.0, 1.0],
    "kappa_1e6": [1.0, 10.0, 1e3, 1e6],
}


def _check(function, inputs, step):
    """
    gradcheck with a finite-difference step relative to the smallest eigenvalue.

    At kappa = 1e6 the finite differences themselves are only accurate to about
    1e-4 (rounding of the decomposition, ~1e-16 kappa, over the step): the
    tolerance is loosened there, and the exact comparison with the autograd
    path (test_*_matches_autograd) covers that conditioning.
    """
    loose = any(t.abs().max() > 1e5 for t in inputs)
    return gradcheck(
        function,
        inputs,
        eps=1e-6 * step,
        atol=1e-3 if loose else 1e-5,
        rtol=1e-2 if loose else 1e-4,
        nondet_tol=0.0,
    )


@pytest.mark.parametrize("spectrum", list(SPECTRA))
def test_gradcheck_transport_to_identity(spectrum):
    source = _spd(SPECTRA[spectrum]).requires_grad_(True)
    tangent = symmetrize(torch.randn(3, N, N, dtype=F64)).requires_grad_(True)
    assert _check(
        lambda v, b: bw.ParallelTransportToIdentityBW.apply(
            symmetrize(v), symmetrize(b)
        ),
        (tangent, source),
        min(SPECTRA[spectrum]),
    )


@pytest.mark.parametrize("spectrum", list(SPECTRA))
def test_transport_to_identity_inverts_from_identity(spectrum):
    point = _spd(SPECTRA[spectrum])
    tangent = symmetrize(torch.randn(3, N, N, dtype=F64))
    back = bw.ParallelTransportFromIdentityBW.apply(
        bw.ParallelTransportToIdentityBW.apply(tangent, point), point
    )
    torch.testing.assert_close(back, tangent)
    torch.testing.assert_close(
        bw.ParallelTransportToIdentityBW.apply(tangent, point),
        bw.bures_wasserstein_parallel_transport_to_identity(tangent, point),
    )


@pytest.mark.parametrize("spectrum", list(SPECTRA))
def test_gradcheck_eigh_clamp(spectrum):
    eigvals = SPECTRA[spectrum]
    data = _spd(eigvals).requires_grad_(True)
    scale = max(eigvals)
    # bounds away from the spectrum (the clamp is not differentiable at a bound)
    lo, hi = 0.5 * min(eigvals) + 0.25, 0.9 * scale
    assert _check(
        lambda x: EighClamp.apply(symmetrize(x), lo, hi) / scale,
        (data,),
        min(eigvals),
    )


def test_eigh_clamp_matches_autograd():
    data = random_spd(5, N, kappa=1e4).requires_grad_(True)
    manual = EighClamp.apply(data, 0.1, 10.0)
    (grad_manual,) = torch.autograd.grad(manual.square().sum(), data)
    auto = eigh_clamp(data, 0.1, 10.0)[0]
    (grad_auto,) = torch.autograd.grad(auto.square().sum(), data)
    torch.testing.assert_close(manual, auto)
    torch.testing.assert_close(grad_manual, grad_auto)


@pytest.mark.parametrize("spectrum", list(SPECTRA))
def test_gradcheck_eigvalsh(spectrum):
    data = _spd(SPECTRA[spectrum]).requires_grad_(True)
    scale = max(SPECTRA[spectrum])
    # symmetric function of the eigenvalues: differentiable at ties
    assert _check(
        lambda x: (EigvalshSymmetric.apply(symmetrize(x)) / scale).square().sum(),
        (data,),
        min(SPECTRA[spectrum]),
    )


@pytest.mark.parametrize("spectrum", list(SPECTRA))
def test_gradcheck_affine_invariant_norm(spectrum):
    base = _spd(SPECTRA[spectrum]).requires_grad_(True)
    scale = max(SPECTRA[spectrum])
    tangent = (scale * symmetrize(torch.randn(N, N, dtype=F64))).requires_grad_(True)
    assert _check(
        lambda b, v: AffineInvariantNorm.apply(symmetrize(b), symmetrize(v)),
        (base, tangent),
        min(SPECTRA[spectrum]),
    )
    torch.testing.assert_close(
        AffineInvariantNorm.apply(base, tangent), affine_invariant_norm(base, tangent)
    )


BW_FUNCTIONS = {
    "log_identity": lambda x, b: bw.BuresWassersteinLogIdentity(x),
    "log": lambda x, b: bw.BuresWassersteinLog(x, b),
    "geodesic": lambda x, b: bw.BuresWassersteinGeodesic(b, x, 0.3),
    "mean": lambda x, b: bw.BuresWassersteinMean(x, n_iterations=2) + 0 * b,
    "center": lambda x, b: bw.BuresWassersteinCenter(x, b),
    "scale": lambda x, b: (
        bw.BuresWassersteinScale(
            x, torch.tensor(0.7, dtype=F64), torch.tensor(1.3, dtype=F64)
        )
        + 0 * b
    ),
    "bias": lambda x, b: bw.BuresWassersteinBias(x, b),
}


@pytest.mark.parametrize("spectrum", list(SPECTRA))
@pytest.mark.parametrize("name", list(BW_FUNCTIONS))
def test_gradcheck_bures_wasserstein_manual(name, spectrum):
    """Manual BW data path: gradcheck at ties, at the identity and kappa = 1e6."""
    eigvals = SPECTRA[spectrum]
    scale = max(eigvals)
    base = _spd(eigvals).requires_grad_(True)
    data = torch.stack([_spd(eigvals, seed=s) for s in (1, 2, 3)])
    data = data.requires_grad_(True)
    function = BW_FUNCTIONS[name]
    assert _check(
        lambda x, b: function(symmetrize(x), symmetrize(b)) / scale,
        (data, base),
        min(eigvals),
    )


@pytest.mark.parametrize("kappa", [1e2, 1e6])
@pytest.mark.parametrize("name", list(BW_FUNCTIONS))
def test_bures_wasserstein_manual_matches_autograd(name, kappa):
    """Same values and gradients as the autograd path on generic inputs."""
    data = random_spd(5, N, kappa=kappa, seed=1).requires_grad_(True)
    base = random_spd(1, N, kappa=kappa, seed=2)[0].requires_grad_(True)
    autograd_functions = {
        "log_identity": lambda x, b: bw.bures_wasserstein_log_identity(x),
        "log": bw.bures_wasserstein_log,
        "geodesic": lambda x, b: bw.bures_wasserstein_geodesic(b, x, 0.3),
        "mean": lambda x, b: bw.bures_wasserstein_mean(x, n_iterations=2) + 0 * b,
        "center": bw.bures_wasserstein_center,
        "scale": lambda x, b: (
            bw.bures_wasserstein_scale(
                x, torch.tensor(0.7, dtype=F64), torch.tensor(1.3, dtype=F64)
            )
            + 0 * b
        ),
        "bias": bw.bures_wasserstein_bias,
    }
    manual = BW_FUNCTIONS[name](data, base)
    auto = autograd_functions[name](data, base)
    torch.testing.assert_close(manual, auto, rtol=1e-9, atol=1e-9 * kappa)
    grads_manual = torch.autograd.grad(
        manual.square().sum(), (data, base), allow_unused=True, materialize_grads=True
    )
    grads_auto = torch.autograd.grad(
        auto.square().sum(), (data, base), allow_unused=True, materialize_grads=True
    )
    for g_manual, g_auto in zip(grads_manual, grads_auto):
        scale = g_auto.abs().max()
        torch.testing.assert_close(g_manual, g_auto, rtol=1e-7, atol=1e-7 * scale)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_gbwbn_scalar_batch_has_finite_gradient(theta):
    """A batch of scalar matrices cI: NaN gradient with autograd, finite manually."""
    scalars = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=F64)
    data = (scalars[:, None, None] * torch.eye(N, dtype=F64)).requires_grad_(True)
    layer = BatchNormSPDMeanScalarVariance(
        N, mean_type="bures_wasserstein", bw_theta=theta
    )
    layer(data).square().sum().backward()
    assert torch.isfinite(data.grad).all()
    for parameter in layer.parameters():
        assert torch.isfinite(parameter.grad).all()

    data_auto = data.detach().clone().requires_grad_(True)
    layer_auto = BatchNormSPDMeanScalarVariance(
        N, mean_type="bures_wasserstein", bw_theta=theta, use_autograd=True
    )
    layer_auto(data_auto).square().sum().backward()
    assert not torch.isfinite(data_auto.grad).all()

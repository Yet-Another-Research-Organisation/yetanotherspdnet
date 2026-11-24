"""ResidualBlock: fused decomposition, eigendecomposition count, options.

- the fused decomposition (FlooredEighFunctions) against the composition it
  replaces (ReEig, then eigvalsh / sqrtm and inverse sqrtm / logm), values and
  gradients, and gradchecks (float64, repeated eigenvalues, active floor);
- the affine-invariant step from one decomposition against the previous
  composition (Cholesky norm, AffineInvariantExp, eigenvalue clamp);
- the number of eigendecompositions of the forward (2 per block) and of the
  backward (0);
- each option: shapes, SPD outputs, gradients, manual against autograd with
  the same weights in training mode, closed forms (eigenbasis, transport,
  congruence, log output), identity at a zero field.
"""

import copy
import math
import warnings

import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_close

from utils import is_spd
from yetanotherspdnet.functions import spd_linalg
from yetanotherspdnet.functions.dispatch import AutogradOnlyWarning
from yetanotherspdnet.functions.spd_geometries import affine_invariant
from yetanotherspdnet.functions.spd_linalg import (
    EighReLu,
    EigvalshSymmetric,
    ExpmSymmetric,
    FlooredEighFunctions,
    LogmSPD,
    SqrtmAndInvSqrtmSPD,
    floored_eigh_functions,
    spd_floor,
)
from yetanotherspdnet.model import GBWBNRResNet, RResNet
from yetanotherspdnet.nn.base import LogEig
from yetanotherspdnet.nn.rresnet_layers import ResidualBlock


F64 = torch.float64
N = 5


@pytest.fixture(autouse=True)
def _no_autograd_only_warning():
    # the default branch BiMap uses torch's orthogonal parametrization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutogradOnlyWarning)
        yield


def random_spd(n_matrices, n=N, kappa=10.0, seed=0, eigvals=None, return_q=False):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(n_matrices, n, n, dtype=F64, generator=generator)
    )
    if eigvals is None:
        u = torch.rand(n_matrices, n, dtype=F64, generator=generator)
        eigvals = kappa ** (u - 0.5)
    else:
        eigvals = torch.as_tensor(eigvals, dtype=F64).expand(n_matrices, n)
    x = (q * eigvals.unsqueeze(-2)) @ q.mT
    x = (x + x.mT) / 2
    return (x, q, eigvals) if return_q else x


def random_sym(n_matrices, n=N, seed=1):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n_matrices, n, n, dtype=F64, generator=generator)
    return (a + a.mT) / 2


def floor(eigvals):
    return spd_floor(eigvals, floor=1e-8)


def block(**kwargs):
    torch.manual_seed(0)
    return ResidualBlock(N, generator=torch.Generator().manual_seed(0), **kwargs)


def pair(**kwargs):
    """Manual and autograd blocks with the same weights."""
    manual = block(use_autograd=False, **kwargs)
    auto = block(use_autograd=True, **kwargs)
    auto.load_state_dict(manual.state_dict())
    return manual, auto


def outputs_and_grads(module, data, cotangent=None):
    module.train()
    data = data.detach().clone().requires_grad_(True)
    out = module(data)
    if cotangent is None:
        cotangent = random_sym(out.shape[0], out.shape[-1], seed=7)
    params = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out, [data, *params], cotangent, allow_unused=True)
    return out.detach(), grads


# ---------------------------------------------------------------------------
# fused decomposition
# ---------------------------------------------------------------------------
REPEATED = [0.5, 0.5, 2.0, 2.0, 2.0]
WITH_FLOORED = [1e-12, 1e-11, 0.3, 1.0, 2.0]  # two eigenvalues below the floor


@pytest.mark.parametrize(
    "eigvals", [None, REPEATED, WITH_FLOORED], ids=["distinct", "repeated", "floored"]
)
def test_floored_eigh_functions_match_the_composition(eigvals):
    X = random_spd(4, eigvals=eigvals)
    cot = [random_sym(4, seed=s) for s in (2, 3, 4)]

    def fused(x):
        lam, _, _, s, r, lg = FlooredEighFunctions.apply(
            x, floor, ("sqrt", "inv_sqrt", "log")
        )
        return lam, s, r, lg

    def composed(x):
        y = EighReLu.apply(x, floor)
        s, r = SqrtmAndInvSqrtmSPD.apply(y)
        return EigvalshSymmetric.apply(y), s, r, LogmSPD.apply(y)

    results = []
    for f in (fused, composed):
        x = X.clone().requires_grad_(True)
        lam, s, r, lg = f(x)
        # a symmetric function of the eigenvalues: individual eigenvalues are
        # not differentiable at ties
        loss = (lam**2 + lam.sin()).sum() + sum(
            (m * c).sum() for m, c in zip((s, r, lg), cot)
        )
        (grad,) = torch.autograd.grad(loss, x)
        results.append(tuple(t.detach() for t in (lam, s, r, lg, grad)))
    for new, old in zip(*results):
        scale = old.abs().max()
        if eigvals is WITH_FLOORED:
            # the composition rounds the floored matrix and decomposes it
            # again: its small eigenvalues (1e-8) carry an error of eps ||X||,
            # 2e-8 relative (2e-8 relative on X^{-1/2})
            assert (new - old).abs().max() <= 1e-7 * float(scale)
        else:
            assert (new - old).abs().max() <= 1e-12 * max(1.0, float(scale))


def test_floored_eigh_functions_exact_on_the_floored_spectrum():
    # the fused decomposition floors the spectrum without the round trip: its
    # X^{-1/2} and log X are exact to rounding, unlike the composition
    X, q, lam = random_spd(4, eigvals=WITH_FLOORED, return_q=True)
    floored = lam.clamp_min(1e-8)
    _, _, _, r, lg = FlooredEighFunctions.apply(X, floor, ("inv_sqrt", "log"))
    exact_r = (q * floored.rsqrt().unsqueeze(-2)) @ q.mT
    exact_log = (q * floored.log().unsqueeze(-2)) @ q.mT
    assert_close(r, exact_r, rtol=1e-12, atol=1e-12 * float(exact_r.abs().max()))
    assert_close(lg, exact_log, rtol=1e-12, atol=1e-12 * float(exact_log.abs().max()))


@pytest.mark.parametrize(
    "eigvals", [None, REPEATED, WITH_FLOORED], ids=["distinct", "repeated", "floored"]
)
@pytest.mark.parametrize("functions", [("sqrt", "inv_sqrt"), ("log",), ()])
def test_floored_eigh_functions_gradcheck(eigvals, functions):
    # floor 0.1 for the floored case: the finite differences (1e-6) must not
    # cross the floor
    if eigvals is WITH_FLOORED:
        eigvals, bound = [0.01, 0.02, 0.3, 1.0, 2.0], 0.1
    else:
        bound = 1e-8
    X = random_spd(2, eigvals=eigvals).requires_grad_(True)

    def f(x):
        outputs = FlooredEighFunctions.apply((x + x.mT) / 2, bound, functions)
        lam = outputs[0]
        # symmetric function of the spectrum (ties: see above)
        return ((lam**2 + lam.sin()).sum(-1), *outputs[3:])

    assert gradcheck(f, (X,))


def test_floored_eigh_functions_autograd_twin():
    X = random_spd(3)
    manual = FlooredEighFunctions.apply(X, floor, ("sqrt", "inv_sqrt", "log"))
    auto = floored_eigh_functions(X, floor, ("sqrt", "inv_sqrt", "log"))
    for a, b in zip(manual, auto):
        assert_close(a, b)


@pytest.mark.parametrize("eigvals", [None, REPEATED], ids=["distinct", "repeated"])
def test_affine_invariant_exp_gradcheck(eigvals):
    X = random_spd(2, eigvals=eigvals).requires_grad_(True)
    V = (0.3 * random_sym(2)).requires_grad_(True)
    assert gradcheck(
        lambda x, v: affine_invariant.AffineInvariantExp.apply(
            (x + x.mT) / 2, (v + v.mT) / 2
        ),
        (X, V),
    )


def _fused_step(x, v):
    """Unit affine-invariant step from one decomposition (as ResidualBlock)."""
    _, _, _, s, r = FlooredEighFunctions.apply(x, floor, ("sqrt", "inv_sqrt"))
    w = spd_linalg.symmetrize(r @ v @ r)
    w = w / torch.linalg.matrix_norm(w).clamp_min(1e-12)[..., None, None]
    return spd_linalg.symmetrize(s @ ExpmSymmetric.apply(w) @ s)


def _previous_step(x, v):
    """The #18 composition: ReEig floor, Cholesky norm, AffineInvariantExp, clamp."""
    y = EighReLu.apply(x, floor)
    norm = affine_invariant.AffineInvariantNorm.apply(y, v).clamp_min(1e-12)
    out = affine_invariant.AffineInvariantExp.apply(y, v / norm[..., None, None])
    return spd_linalg.EighClamp.apply(spd_linalg.symmetrize(out), floor, 1e8)


@pytest.mark.parametrize("eigvals", [None, REPEATED], ids=["distinct", "repeated"])
def test_fused_step_matches_previous_composition(eigvals):
    X = random_spd(4, eigvals=eigvals)
    V = random_sym(4)
    cot = random_sym(4, seed=9)
    results = []
    for f in (_fused_step, _previous_step):
        x = X.clone().requires_grad_(True)
        v = V.clone().requires_grad_(True)
        out = f(x, v)
        grads = torch.autograd.grad(out, (x, v), cot)
        results.append(tuple(t.detach() for t in (out, *grads)))
    for new, old in zip(*results):
        assert (new - old).abs().max() <= 1e-12 * max(1.0, float(old.abs().max()))


@pytest.mark.parametrize("eigvals", [None, REPEATED], ids=["distinct", "repeated"])
def test_fused_step_gradcheck(eigvals):
    # the (X, V) -> X_+ map, V an independent input: differentiable at repeated
    # eigenvalues of X (the full block is not: f acts position-wise on the
    # sorted spectrum)
    X = random_spd(2, eigvals=eigvals).requires_grad_(True)
    V = random_sym(2).requires_grad_(True)
    assert gradcheck(lambda x, v: _fused_step((x + x.mT) / 2, (v + v.mT) / 2), (X, V))


# ---------------------------------------------------------------------------
# number of eigendecompositions
# ---------------------------------------------------------------------------
class _Counter:
    def __init__(self, monkeypatch):
        self.counts = {"eigh": 0, "eigvalsh": 0, "cholesky": 0}
        for name in self.counts:
            original = getattr(torch.linalg, name)

            def wrapped(*args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(torch.linalg, name, wrapped)

    def take(self):
        counts = dict(self.counts)
        for name in self.counts:
            self.counts[name] = 0
        return counts


# (options, eigh + eigvalsh calls of one forward); the default branch
# BiMap -> ReEig has one eigh (ReEig)
EIGH_COUNTS = [
    ({}, 2),
    ({"metric": "log_euclidean"}, 2),
    ({"metric": "log_euclidean", "le_output": "log"}, 1),
    ({"basis": "eigenbasis"}, 1),
    ({"metric": "log_euclidean", "basis": "eigenbasis"}, 1),
    ({"output_clamp": True}, 3),
    ({"step": "clip"}, 2),
    ({"step": "none"}, 3),  # automatic output clamp
    ({"step": "learned_scale", "output_clamp": False}, 2),
    ({"residual": "branch_pt"}, 1 + 1 + 1 + 1 + 1),  # X, ReEig, Z, N^{-1/2}, exp
    ({"metric": "log_euclidean", "residual": "branch_pt"}, 1 + 1 + 1 + 1),
    ({"residual": "congruence"}, 1 + 1),  # X^{1/2}, ReEig
]


@pytest.mark.parametrize(
    ("options", "expected"), EIGH_COUNTS, ids=[str(o) for o, _ in EIGH_COUNTS]
)
def test_eigh_count(options, expected, monkeypatch):
    layer = block(**options)
    layer.train()
    X = random_spd(8).requires_grad_(True)
    counter = _Counter(monkeypatch)
    out = layer(X)
    forward = counter.take()
    out.square().sum().backward()
    backward = counter.take()
    assert forward["eigh"] + forward["eigvalsh"] == expected
    assert forward["cholesky"] == 0
    assert sum(backward.values()) == 0
    assert torch.isfinite(X.grad).all()


def test_eigh_count_models(monkeypatch):
    model = RResNet(N, [4], [3], 2, generator=torch.Generator().manual_seed(0))
    counter = _Counter(monkeypatch)
    model(random_spd(8))
    # BiMap (no decomposition), 3 blocks x 2, LogEig 1
    assert counter.take()["eigh"] == 3 * 2 + 1
    model = RResNet(
        N,
        [4],
        [3],
        2,
        residual_metric="log_euclidean",
        residual_options={"le_output": "log"},
        generator=torch.Generator().manual_seed(0),
    )
    model(random_spd(8))
    # 2 + 2 + 1 (log output), no LogEig
    assert counter.take()["eigh"] == 5


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
OPTIONS = [
    {},
    {"metric": "log_euclidean"},
    {"metric": "log_euclidean", "le_output": "log"},
    {"step": "none"},
    {"step": "clip", "step_radius": 0.5},
    {"step": "learned_scale"},
    {"metric": "log_euclidean", "step": "unit"},
    {"metric": "log_euclidean", "step": "learned_scale"},
    {"spectrum_input": "log"},
    {"basis": "eigenbasis"},
    {"metric": "log_euclidean", "basis": "eigenbasis"},
    {"metric": "log_euclidean", "basis": "eigenbasis", "le_output": "log"},
    {"field_batchnorm": False},
    {"output_clamp": True},
    {"residual": "branch_pt"},
    {"residual": "branch_pt", "basis": "eigenbasis"},
    {"metric": "log_euclidean", "residual": "branch_pt"},
    {"residual": "congruence"},
    {"spectrum_type": "mlp", "basis": "eigenbasis", "step": "clip"},
]
IDS = [",".join(f"{k}={v}" for k, v in o.items()) or "default" for o in OPTIONS]


@pytest.mark.parametrize("options", OPTIONS, ids=IDS)
def test_options_shapes_spd_and_gradients(options):
    layer = block(**options)
    X = random_spd(6, kappa=1e3)
    out, grads = outputs_and_grads(layer, X)
    assert out.shape == X.shape
    assert torch.isfinite(out).all()
    if options.get("le_output") == "log":
        assert torch.allclose(out, out.mT)
    else:
        assert is_spd(out)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().max() > 0
    layer.eval()
    with torch.no_grad():
        assert torch.isfinite(layer(X)).all()


@pytest.mark.parametrize("options", OPTIONS, ids=IDS)
def test_options_manual_matches_autograd(options):
    # same weights, training mode: outputs, input and parameter gradients
    manual, auto = pair(**options)
    X = random_spd(6, kappa=1e2, seed=3)
    out_m, grads_m = outputs_and_grads(manual, X)
    out_a, grads_a = outputs_and_grads(auto, X)
    assert_close(out_m, out_a, rtol=1e-10, atol=1e-12)
    for g_m, g_a in zip(grads_m, grads_a):
        if g_m is None or g_a is None:
            assert g_m is None and g_a is None
            continue
        assert_close(g_m, g_a, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"metric": "log_euclidean"},
        {"basis": "eigenbasis"},
        {"metric": "log_euclidean", "basis": "eigenbasis"},
        {"residual": "branch_pt"},
        {"residual": "congruence"},
        {"step": "learned_scale"},
    ],
    ids=[
        "AI",
        "LE",
        "AI-eigenbasis",
        "LE-eigenbasis",
        "branch_pt",
        "congruence",
        "learned",
    ],
)
def test_block_gradcheck(options):
    # whole block on distinct spectra, in evaluation mode (BatchNorm1d of the
    # field with running statistics, so that the map is deterministic)
    layer = block(**options)
    layer.eval()
    X = random_spd(2, seed=4).requires_grad_(True)
    assert gradcheck(lambda x: layer((x + x.mT) / 2), (X,))


def _zero_field(layer):
    with torch.no_grad():
        last = layer.vector_field.spectrum_map[-1]
        last.weight.zero_()
        last.bias.zero_()


@pytest.mark.parametrize(
    "options",
    [
        {"step": "none"},
        {"step": "learned_scale"},
        {"step": "clip"},
        {"metric": "log_euclidean"},
        {"basis": "eigenbasis", "step": "learned_scale"},
        {"metric": "log_euclidean", "basis": "eigenbasis"},
    ],
)
def test_identity_for_a_zero_field(options):
    layer = block(**options)
    _zero_field(layer)
    X = random_spd(4, kappa=1e2)
    assert_close(layer(X), X, rtol=1e-12, atol=1e-13)


def test_learned_scale_starts_small():
    layer = block(step="learned_scale", step_scale_init=1e-2)
    assert layer.step_scale.item() == pytest.approx(1e-2)
    X = random_spd(4, kappa=1e2)
    out = layer(X)
    # affine-invariant distance = alpha
    distance = torch.linalg.matrix_norm(
        _whiten(X, affine_invariant.affine_invariant_log(X, out))
    )
    assert_close(distance, torch.full_like(distance, 1e-2), rtol=1e-8, atol=0)


def _whiten(x, v):
    _, _, _, _, r = floored_eigh_functions(x, floor, ("sqrt", "inv_sqrt"))
    return r @ v @ r


@pytest.mark.parametrize(
    ("step", "expected"),
    [("unit", lambda nu: torch.ones_like(nu)), ("clip", lambda nu: nu.clamp(max=0.3))],
)
def test_step_length(step, expected):
    # affine-invariant length of the step: 1 (unit), min(||V||_X, r) (clip)
    layer = block(step=step, step_radius=0.3)
    layer.eval()
    X = random_spd(6, kappa=1e2)
    with torch.no_grad():
        f = layer.vector_field.spectrum(torch.linalg.eigvalsh(X))
        V = layer.vector_field.from_spectrum(f)
        nu = torch.linalg.matrix_norm(_whiten(X, V))
        out = layer(X)
    length = torch.linalg.matrix_norm(
        _whiten(X, affine_invariant.affine_invariant_log(X, out))
    )
    assert_close(length, expected(nu), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("radius", [1.0, 0.3])
def test_unit_step_spectrum_bound(radius):
    # lambda(X_+) in [lambda_min e^{-r}, lambda_max e^{r}]: the eigenvalue clamp
    # of the reference (projx) is inactive, so it is off by default
    step = "unit" if radius == 1.0 else "clip"
    layer = block(step=step, step_radius=radius)
    X = random_spd(32, kappa=1e6, seed=5)
    lam = torch.linalg.eigvalsh(X)
    out = layer(X).detach()
    lam_out = torch.linalg.eigvalsh(out)
    tol = 1e-10
    assert (
        lam_out.min(-1).values >= lam.min(-1).values * math.exp(-radius) * (1 - tol)
    ).all()
    assert (
        lam_out.max(-1).values <= lam.max(-1).values * math.exp(radius) * (1 + tol)
    ).all()
    clamped = block(step=step, step_radius=radius, output_clamp=True)
    assert_close(clamped(X), out, rtol=1e-12, atol=1e-14)


def test_eigenbasis_closed_form_affine_invariant():
    # Exp_X(V / ||V||_X) with V = U diag(f) U^T on the eigenvectors of X
    layer = block(basis="eigenbasis")
    layer.eval()
    X = random_spd(4, kappa=1e2)
    with torch.no_grad():
        lam, U = torch.linalg.eigh(X)
        V = (U * layer.vector_field.spectrum(lam).unsqueeze(-2)) @ U.mT
        nu = affine_invariant.affine_invariant_norm(X, V)
        expected = affine_invariant.affine_invariant_exp(X, V / nu[..., None, None])
        assert_close(layer(X), expected, rtol=1e-10, atol=1e-12)


def test_eigenbasis_closed_form_log_euclidean():
    layer = block(basis="eigenbasis", metric="log_euclidean")
    layer.eval()
    X = random_spd(4, kappa=1e2)
    with torch.no_grad():
        lam, U = torch.linalg.eigh(X)
        V = (U * layer.vector_field.spectrum(lam).unsqueeze(-2)) @ U.mT
        expected = torch.linalg.matrix_exp(spd_linalg.logm_SPD(X)[0] + V)
        assert_close(layer(X), expected, rtol=1e-10, atol=1e-12)


def test_branch_pt_closed_form():
    # Exp_X(Gamma_{Z -> X} V(Z) / ||.||_X), with the batched parallel transport
    layer = block(residual="branch_pt")
    layer.eval()
    X = random_spd(4, kappa=1e2)
    with torch.no_grad():
        Z = layer.branch(X)
        V = layer.vector_field(Z)
        transported = affine_invariant.affine_invariant_parallel_transport(V, Z, X)
        nu = affine_invariant.affine_invariant_norm(X, transported)
        # the transport is an isometry: normalizing at Z or at X is the same
        assert_close(nu, affine_invariant.affine_invariant_norm(Z, V))
        expected = affine_invariant.affine_invariant_exp(
            X, transported / nu[..., None, None]
        )
        assert_close(layer(X), expected, rtol=1e-10, atol=1e-12)


def test_congruence_closed_form():
    # X^{1/2} Z X^{1/2} = congruence_SPD_sqrtm(Z, X) = Exp_X(Gamma_{I -> X} log Z)
    layer = block(residual="congruence")
    layer.eval()
    X = random_spd(4, kappa=1e2)
    with torch.no_grad():
        Z = layer.branch(X)
        out = layer(X)
        assert_close(out, spd_linalg.congruence_SPD_sqrtm(Z, X), rtol=1e-10, atol=1e-12)
        log_z = spd_linalg.logm_SPD(Z)[0]
        identity = torch.eye(N, dtype=F64).expand_as(X)
        transported = affine_invariant.affine_invariant_parallel_transport(
            log_z, identity, X
        )
        expected = affine_invariant.affine_invariant_exp(X, transported)
        assert_close(out, expected, rtol=1e-10, atol=1e-12)


def test_le_output_log_is_the_logarithm():
    spd = block(metric="log_euclidean")
    log = block(metric="log_euclidean", le_output="log")
    log.load_state_dict(spd.state_dict())
    X = random_spd(4, kappa=1e2)
    assert_close(log(X), spd_linalg.logm_SPD(spd(X))[0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("model_cls", "args"),
    [(RResNet, (N, [4, 3], [1, 2], 2)), (GBWBNRResNet, (N, 4, 2))],
    ids=["RResNet", "GBWBNRResNet"],
)
def test_models_le_output_log_skips_logeig(model_cls, args):
    common = {
        "residual_metric": "log_euclidean",
        "generator": torch.Generator().manual_seed(0),
    }
    torch.manual_seed(0)
    spd = model_cls(*args, **common)
    torch.manual_seed(0)
    common["generator"] = torch.Generator().manual_seed(0)
    log = model_cls(*args, residual_options={"le_output": "log"}, **common)
    log.load_state_dict(spd.state_dict())
    assert isinstance(spd.spd_layers[-1], LogEig)
    assert isinstance(log.spd_layers[-1], ResidualBlock)
    assert log.spd_layers[-1].le_output == "log"
    # only the last block returns log coordinates
    blocks = [m for m in log.spd_layers if isinstance(m, ResidualBlock)]
    assert [b.le_output for b in blocks] == ["spd"] * (len(blocks) - 1) + ["log"]
    X = random_spd(8)
    spd.train()
    log.train()
    assert_close(log(X), spd(X), rtol=1e-9, atol=1e-11)
    assert "residual_options" in repr(log) and "residual_options" not in repr(spd)


def test_model_residual_options_are_forwarded():
    model = RResNet(
        N,
        [4],
        [2],
        2,
        residual_options={"step": "clip", "basis": "eigenbasis"},
        generator=torch.Generator().manual_seed(0),
    )
    blocks = [m for m in model.spd_layers if isinstance(m, ResidualBlock)]
    assert all(
        b.step == "clip" and b.vector_field.basis == "eigenbasis" for b in blocks
    )
    model(random_spd(4)).sum().backward()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": "big"}, "step"),
        ({"residual": "sum"}, "residual"),
        ({"le_output": "log"}, "le_output"),
        ({"residual": "congruence", "step": "unit"}, "congruence"),
        (
            {"residual": "congruence", "metric": "log_euclidean", "le_output": "log"},
            "le_output",
        ),
        ({"basis": "U"}, "basis"),
        ({"spectrum_input": "sqrt"}, "spectrum_input"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ResidualBlock(N, **kwargs)


def test_model_le_output_log_validation():
    with pytest.raises(ValueError, match="last stage"):
        RResNet(
            N,
            [4, 3],
            [1, 0],
            2,
            residual_options={"le_output": "log"},
            residual_metric="log_euclidean",
        )
    with pytest.raises(ValueError, match="use_logeig"):
        GBWBNRResNet(
            N,
            4,
            2,
            use_logeig=False,
            residual_options={"le_output": "log"},
            residual_metric="log_euclidean",
        )


def test_default_repr_and_hash_unchanged():
    # the options only appear in the representation (and the model hash) when set
    assert "step" not in repr(block()) and "residual=" not in repr(block())
    assert "step=clip" in repr(block(step="clip"))


def test_block_dynamic_hook_registration():
    layer = block(
        stiefel_parametrization_mode="dynamic",
        stiefel_n_steps_ref_update=1,
        residual="branch_pt",
    )
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer.register_optimizer_hook(optimizer)
    layer.train()
    layer(random_spd(4)).square().sum().backward()
    optimizer.step()
    assert layer.vector_field.current_ref_step == 0
    original = layer.vector_field.parametrizations.Q.original
    assert_close(original, torch.zeros_like(original))


def test_deepcopy_and_state_dict_roundtrip():
    layer = block(step="learned_scale", residual="branch_pt")
    clone = copy.deepcopy(layer)
    X = random_spd(3)
    layer.eval()
    clone.eval()
    assert_close(layer(X), clone(X))


def test_model_branch_factory_one_module_per_block():
    from yetanotherspdnet.nn.base import BiMap, ReEig

    def make_branch(n_features):
        return torch.nn.Sequential(
            BiMap(n_features, n_features, parametrization_mode="dynamic"), ReEig()
        )

    model = RResNet(
        N,
        [4, 3],
        [2, 2],
        2,
        residual_options={"residual": "branch_pt", "branch": make_branch},
        generator=torch.Generator().manual_seed(0),
    )
    blocks = [m for m in model.spd_layers if isinstance(m, ResidualBlock)]
    branches = [b.branch for b in blocks]
    assert len({id(b) for b in branches}) == 4
    assert [b[0].n_in for b in branches] == [4, 4, 3, 3]
    model(random_spd(4)).sum().backward()
    # the factory appears by name in the representation (stable model hash)
    assert "make_branch" in repr(model) and "0x" not in repr(model)
    with pytest.raises(ValueError, match="factory"):
        RResNet(N, [4], [1], 2, residual_options={"branch": make_branch(4)})

"""dtype-aware floors, guarded standard deviations."""

import warnings

import pytest
import torch

from yetanotherspdnet.functions.dispatch import AutogradOnlyWarning
from yetanotherspdnet.functions.scalar_functions import std_from_variance
from yetanotherspdnet.functions.spd_geometries import (
    affine_invariant,
    bures_wasserstein,
    kullback_leibler,
    kullback_leibler_symmetrized,
    log_euclidean,
)
from yetanotherspdnet.functions.spd_linalg import EighReLu, eigh_relu, spd_floor
from yetanotherspdnet.nn import BatchNormSPDMeanScalarVariance, ResidualBlock


F64 = torch.float64
N = 4


def test_spd_floor_dtype():
    assert spd_floor(torch.float64) == 1e-8
    assert spd_floor(torch.float32) == pytest.approx(torch.finfo(torch.float32).eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spd_floor_eigvals(dtype):
    eigvals = torch.tensor([[1e-9, 1.0, 1e2], [1e-3, 1.0, 1e7]], dtype=dtype)
    floor = spd_floor(eigvals)
    assert floor.shape == (2, 1)
    eps = torch.finfo(dtype).eps
    expected = torch.clamp(eps * eigvals.amax(-1, keepdim=True), min=1e-8)
    torch.testing.assert_close(floor, expected)
    if dtype == torch.float64:
        # unchanged absolute floor at moderate scale
        assert floor[0].item() == 1e-8


def test_eigh_relu_callable_floor():
    data = torch.diag(torch.tensor([1e-12, 1.0, 1e9], dtype=F64))
    out, _, _ = eigh_relu(data, spd_floor)
    expected_floor = torch.finfo(F64).eps * 1e9
    assert torch.linalg.eigvalsh(out)[0].item() == pytest.approx(expected_floor)
    torch.testing.assert_close(EighReLu.apply(data, spd_floor), out)


VARIANCE_FUNCTIONS = {
    "affine_invariant": affine_invariant.affine_invariant_std_scalar,
    "log_euclidean": log_euclidean.log_euclidean_std_scalar,
    "left_kl": kullback_leibler.left_kullback_leibler_std_scalar,
    "right_kl": kullback_leibler.right_kullback_leibler_std_scalar,
    "symmetrized_kl": kullback_leibler_symmetrized.symmetrized_kullback_leibler_std_scalar,
    "bures_wasserstein": bures_wasserstein.bures_wasserstein_std_scalar,
}


def test_std_from_variance():
    variance = torch.tensor([-1e-18, 0.0, 4.0], dtype=F64, requires_grad=True)
    std = std_from_variance(variance)
    torch.testing.assert_close(std, torch.tensor([0.0, 0.0, 2.0], dtype=F64))
    std.sum().backward()
    torch.testing.assert_close(variance.grad, torch.tensor([0.0, 0.0, 0.25], dtype=F64))


@pytest.mark.parametrize("name", list(VARIANCE_FUNCTIONS))
def test_autograd_std_identical_matrices_is_finite(name):
    """Autograd std of identical matrices (std 0): zero subgradient, not NaN."""
    # distinct eigenvalues: autograd through eigh is defined
    generator = torch.Generator().manual_seed(1)
    a = torch.randn(N, N, dtype=F64, generator=generator)
    point = a @ a.T + N * torch.eye(N, dtype=F64)
    data = point.expand(5, N, N).clone().requires_grad_(True)
    reference = point.clone().requires_grad_(True)
    std = VARIANCE_FUNCTIONS[name](data, reference)
    assert std.item() == pytest.approx(0.0, abs=1e-7)
    std.backward()
    assert torch.isfinite(data.grad).all()
    assert torch.isfinite(reference.grad).all()


@pytest.mark.parametrize(
    "mean_type", ["affine_invariant", "log_euclidean", "arithmetic", "harmonic"]
)
@pytest.mark.parametrize("use_autograd", [False, True])
def test_scalar_variance_identical_matrices(mean_type, use_autograd):
    """A batch of identical matrices: finite output and gradients (std floor)."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(N, N, dtype=F64, generator=generator)
    point = a @ a.T + N * torch.eye(N, dtype=F64)
    data = point.expand(4, N, N).clone().requires_grad_(True)
    layer = BatchNormSPDMeanScalarVariance(
        N, mean_type=mean_type, use_autograd=use_autograd
    )
    out = layer(data)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert torch.isfinite(data.grad).all()


def test_residual_block_floor_is_dtype_aware():
    """float32: the input floor rises to the rounding level of the spectrum."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutogradOnlyWarning)
        block = ResidualBlock(N, dtype=torch.float32)
    lam = torch.tensor([1e-9, 1e-2, 1.0, 1e3], dtype=torch.float32)
    q, _ = torch.linalg.qr(torch.randn(N, N))
    data = ((q * lam) @ q.T).expand(3, N, N).clone()
    floored = block.floor(data)
    eigvals = torch.linalg.eigvalsh(floored.double())
    floor = torch.finfo(torch.float32).eps * 1e3
    assert eigvals.min().item() >= 0.5 * floor


def test_per_matrix_floor_backward():
    """Per-matrix (callable) bounds broadcast in the Daleckii-Krein backward."""
    from torch.autograd import gradcheck

    from yetanotherspdnet.functions.spd_linalg import EighClamp, symmetrize

    generator = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(3, N, N, dtype=F64, generator=generator))
    lam = torch.tensor([[1e-12, 0.3, 1.0, 2.0]], dtype=F64) * torch.tensor(
        [[1.0], [1e6], [1e9]], dtype=F64
    )
    data = ((q * lam.unsqueeze(-2)) @ q.mT).requires_grad_(True)
    floor = lambda e: spd_floor(e, floor=1e-6)
    assert gradcheck(
        lambda x: EighReLu.apply(symmetrize(x), floor) / lam.amax(-1)[:, None, None],
        (data,),
        eps=1e-9,
        atol=1e-4,
        rtol=1e-3,
    )
    out = EighClamp.apply(data, floor, None)
    torch.testing.assert_close(out, EighReLu.apply(data, floor))


# ---------------------------------------------------------------------------
# mixed precision: eigendecompositions in float64 for float32 data
# ---------------------------------------------------------------------------
def test_eigh_precision_context():
    from yetanotherspdnet.functions.spd_linalg import (
        eigh_context,
        eigh_precision,
        get_eigh_dtype,
        symmetric_eigh,
    )

    data = torch.diag(torch.tensor([1e-6, 1.0, 1e2]))
    assert get_eigh_dtype() is None
    with eigh_precision(F64):
        assert get_eigh_dtype() == F64
        eigvals, eigvecs = symmetric_eigh(data)
        assert eigvals.dtype == eigvecs.dtype == torch.float32
        with eigh_context(None):
            assert get_eigh_dtype() == F64
        with eigh_precision(None):
            assert get_eigh_dtype() is None
    assert get_eigh_dtype() is None


def _ill_conditioned(dtype, kappa=1e6, n_matrices=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(n_matrices, N, N, dtype=F64, generator=generator)
    )
    lam = kappa ** torch.linspace(-0.5, 0.5, N, dtype=F64)
    return ((q * lam) @ q.mT).to(dtype)


def test_logeig_mixed_precision_is_more_accurate():
    """float32 LogEig at kappa = 1e6: eigh_dtype=float64 removes the eigh error."""
    from yetanotherspdnet.nn import LogEig

    x32 = _ill_conditioned(torch.float32)
    reference = LogEig()(x32.double())  # float64 on the same rounded input
    err32 = (LogEig()(x32).double() - reference).abs().max()
    err_mixed = (LogEig(eigh_dtype=F64)(x32).double() - reference).abs().max()
    assert LogEig(eigh_dtype=F64)(x32).dtype == torch.float32
    assert err_mixed < 1e-5
    assert err_mixed < err32


def test_model_eigh_dtype_keeps_hash_when_unset():
    from yetanotherspdnet.model import SPDnet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutogradOnlyWarning)
        plain = SPDnet(N, [3], 2, dtype=torch.float32)
        mixed = SPDnet(N, [3], 2, dtype=torch.float32, eigh_dtype=F64)
    assert "eigh_dtype" not in repr(plain)
    assert "eigh_dtype=torch.float64" in repr(mixed)
    out = mixed(_ill_conditioned(torch.float32))
    assert out.dtype == torch.float32
    out.sum().backward()

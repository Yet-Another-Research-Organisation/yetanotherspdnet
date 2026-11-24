"""Divided differences of the Daleckii-Krein backward (stable kernels, ties)."""

from decimal import Decimal, getcontext

import pytest
import torch
from torch.autograd import gradcheck

from yetanotherspdnet.functions import scalar_functions as sf
from yetanotherspdnet.functions.spd_linalg import (
    ExpmSymmetric,
    LogmSPD,
    PowmSPD,
    ScaledSoftPlusSymmetric,
    SqrtmSPD,
    _aux_eigh_operation_grad,
    symmetrize,
)


getcontext().prec = 50


def _reference(name: str, a: float, b: float, exponent: float = 0.3) -> float:
    """50-digit divided difference of the exactly represented floats a != b."""
    da, db = Decimal(a), Decimal(b)
    funcs = {
        "sqrt": lambda x: x.sqrt(),
        "inv_sqrt": lambda x: 1 / x.sqrt(),
        "log": lambda x: x.ln(),
        "exp": lambda x: x.exp(),
        "pow": lambda x: (Decimal(exponent) * x.ln()).exp(),
        "inv": lambda x: 1 / x,
    }
    f = funcs[name]
    return float((f(da) - f(db)) / (da - db))


OPERATIONS = {
    "sqrt": (torch.sqrt, sf.sqrt_derivative, None),
    "inv_sqrt": (sf.inv_sqrt, sf.inv_sqrt_derivative, None),
    "log": (torch.log, sf.inv, None),
    "exp": (torch.exp, torch.exp, None),
    "inv": (sf.inv, sf.inv_derivative, None),
    "pow": (
        lambda x: torch.pow(x, 0.3),
        lambda x: 0.3 * torch.pow(x, -0.7),
        lambda a, b: sf.pow_divided_difference(a, b, 0.3),
    ),
}

# (a, b) pairs: the small close pair of the audit (1e-7, 5e-7, relative error
# 10.7 % with the former absolute 1e-6 threshold), a near tie, a wide gap
PAIRS = [(1e-7, 5e-7), (1.0, 1.0 + 1e-9), (2.0, 2.0 + 3e-5), (1e-3, 7.0)]


@pytest.mark.parametrize("name", list(OPERATIONS))
@pytest.mark.parametrize("pair", PAIRS)
def test_divided_difference_float64(name, pair):
    """Off-diagonal Loewner entries agree with a 50-digit reference."""
    a, b = pair
    if name == "exp":
        a, b = 10 * a, 10 * b
    operation, deriv, dd = OPERATIONS[name]
    eigvals = torch.tensor([a, b], dtype=torch.float64)
    loewner = _aux_eigh_operation_grad(eigvals, operation, deriv, dd)
    ref = _reference(name, a, b)
    assert loewner[0, 1].item() == loewner[1, 0].item()
    assert abs(loewner[0, 1].item() - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("name", ["sqrt", "pow"])
def test_audit_case_through_backward(name):
    """SqrtmSPD / PowmSPD backward on diag(1e-7, 5e-7, 1): relative error < 1e-10."""
    lam = torch.tensor([1e-7, 5e-7, 1.0], dtype=torch.float64)
    data = torch.diag(lam).requires_grad_(True)
    if name == "sqrt":
        out = SqrtmSPD.apply(data)
    else:
        out = PowmSPD.apply(data, torch.tensor(0.3, dtype=torch.float64))
    grad_out = torch.zeros(3, 3, dtype=torch.float64)
    grad_out[0, 1] = grad_out[1, 0] = 1.0
    (grad,) = torch.autograd.grad(out, data, grad_out)
    ref = _reference(name, 1e-7, 5e-7)
    assert abs(grad[0, 1].item() - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_divided_difference_float32(name):
    """float32 kernels stay at a few ulps on close pairs."""
    a, b = 1e-3, 1e-3 * (1 + 3e-6)
    operation, deriv, dd = OPERATIONS[name]
    eigvals = torch.tensor([a, b], dtype=torch.float32)
    a32, b32 = (float(v) for v in eigvals)
    loewner = _aux_eigh_operation_grad(eigvals, operation, deriv, dd)
    ref = _reference(name, a32, b32) if a32 != b32 else float(deriv(eigvals[:1]))
    assert abs(loewner[0, 1].item() - ref) <= 1e-5 * abs(ref)


def test_generic_quotient_near_tie():
    """The generic path (no closed form) switches to f'(midpoint) near ties."""
    operation = lambda x: torch.sqrt(x)  # not recognized: generic path
    for gap in (1e-12, 1e-9, 1e-6, 1e-4):
        eigvals = torch.tensor([1.0, 1.0 + gap], dtype=torch.float64)
        loewner = _aux_eigh_operation_grad(eigvals, operation, sf.sqrt_derivative)
        ref = 1 / (1 + (1 + gap) ** 0.5)
        assert abs(loewner[0, 1].item() - ref) <= 1e-10 * ref
        assert loewner[0, 1].item() == loewner[1, 0].item()


@pytest.mark.parametrize(
    "function",
    [ExpmSymmetric.apply, ScaledSoftPlusSymmetric.apply],
    ids=["expm", "softplus"],
)
def test_zero_spectrum_is_finite(function):
    """A zero tangent (SPD parametrization at init) gives a finite gradient."""
    data = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
    function(data).sum().backward()
    assert torch.isfinite(data.grad).all()


def _spd_with_spectrum(eigvals: torch.Tensor, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    n = eigvals.shape[-1]
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    return (q * eigvals) @ q.T


SPECTRA = {
    "repeated": [2.0, 2.0, 2.0, 5.0],
    "nearly_repeated": [2.0, 2.0 + 1e-4, 2.0 + 2e-4, 5.0],
    "kappa_1e6": [1.0, 10.0, 1e3, 1e6],
}


@pytest.mark.parametrize("spectrum", list(SPECTRA))
@pytest.mark.parametrize(
    "function",
    [
        SqrtmSPD.apply,
        LogmSPD.apply,
        lambda x: PowmSPD.apply(x, torch.tensor(0.3, dtype=x.dtype)),
    ],
    ids=["sqrtm", "logm", "powm"],
)
def test_gradcheck_spectra(function, spectrum):
    """gradcheck with repeated, nearly repeated eigenvalues and kappa = 1e6."""
    eigvals = torch.tensor(SPECTRA[spectrum], dtype=torch.float64)
    data = _spd_with_spectrum(eigvals).requires_grad_(True)
    scale = eigvals.max()
    # eigh reads one triangle: symmetrize inside the checked function
    assert gradcheck(
        lambda x: function(symmetrize(x)) / scale.sqrt(),
        (data,),
        eps=1e-6 * scale.item() ** 0.5,
        atol=1e-6,
        rtol=1e-4,
    )

"""float32 layers against a float64 reference, kappa <= 1e4.

Each layer runs forward + backward in float32 and in float64 on the same
float32-rounded input, with the same parameters. The tolerances are about 10x
the largest relative errors measured by the precision benchmark (n = 16 and
64, 8 seeds, see the numerics guide) at kappa <= 1e4; "mixed" is the float32
layer with ``eigh_dtype=torch.float64``.
"""

import copy
import warnings

import pytest
import torch

from yetanotherspdnet.functions.dispatch import AutogradOnlyWarning
from yetanotherspdnet.model import GBWBNRResNet, RResNet, SPDnet
from yetanotherspdnet.nn import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
    BiMap,
    LogEig,
    ReEig,
    ResidualBlock,
)


F32, F64 = torch.float32, torch.float64
N, BATCH = 8, 16
KAPPAS = [1e1, 1e2, 1e4]
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

# name -> (builder(dtype), forward tolerance, backward tolerance)
LAYERS = {
    "ReEig": (lambda dt: ReEig(eps=1e-4), 1e-5, 1e-5),
    "LogEig": (lambda dt: LogEig(), 1e-4, 1e-3),
    "BiMap": (lambda dt: BiMap(N, 4, dtype=dt), 1e-5, 1e-5),
    "ResidualBlock[affine_invariant]": (
        lambda dt: ResidualBlock(
            N, dtype=dt, generator=torch.Generator().manual_seed(0)
        ),
        1e-4,
        3e-3,
    ),
    "ResidualBlock[log_euclidean]": (
        lambda dt: ResidualBlock(
            N,
            metric="log_euclidean",
            dtype=dt,
            generator=torch.Generator().manual_seed(0),
        ),
        1e-4,
        2e-2,
    ),
    "SPDnet": (
        lambda dt: SPDnet(
            N,
            [6, 4],
            3,
            batchnorm=True,
            dtype=dt,
            generator=torch.Generator().manual_seed(0),
        ),
        1e-4,
        1e-3,
    ),
    "RResNet": (
        lambda dt: RResNet(
            N, [6], [1], 3, dtype=dt, generator=torch.Generator().manual_seed(0)
        ),
        2e-4,
        3e-3,
    ),
    "GBWBNRResNet": (
        lambda dt: GBWBNRResNet(
            N, 6, 3, dtype=dt, generator=torch.Generator().manual_seed(0)
        ),
        3e-4,
        5e-2,
    ),
}
for _mt in MEAN_TYPES:
    LAYERS[f"BatchNormSPDMean[{_mt}]"] = (
        lambda dt, mt=_mt: BatchNormSPDMean(N, mean_type=mt, dtype=dt),
        2e-4,
        5e-3,
    )
    LAYERS[f"BatchNormSPDMeanScalarVariance[{_mt}]"] = (
        lambda dt, mt=_mt: BatchNormSPDMeanScalarVariance(N, mean_type=mt, dtype=dt),
        2e-4,
        5e-2 if _mt == "bures_wasserstein" else 5e-3,
    )


# largest kappa tested per layer (default 1e4). The Bures-Wasserstein layers
# were limited to 1e2 while their cross term went through an eigendecomposition
# at kappa^2; the SVD cross term (polar factor) brings them to 1e4, in float32
# and in mixed precision.
MAX_KAPPA: dict[str, float] = {}


def _spd(kappa: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(BATCH, N, N, dtype=F64, generator=generator))
    u = torch.rand(BATCH, N, dtype=F64, generator=generator)
    u[:, 0], u[:, 1] = 0.0, 1.0
    lam = kappa ** (u - 0.5)
    x = (q * lam.unsqueeze(-2)) @ q.mT
    return ((x + x.mT) / 2).to(F32)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.double() - b) / torch.linalg.vector_norm(b))


def _run(module: torch.nn.Module, data: torch.Tensor, cotangent: torch.Tensor):
    data = data.detach().clone().requires_grad_(True)
    out = module(data)
    (grad,) = torch.autograd.grad(out, data, cotangent.to(out.dtype))
    return out.detach(), grad


@pytest.mark.parametrize("mixed", [False, True], ids=["float32", "mixed"])
@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("name", list(LAYERS))
def test_float32_matches_float64(name, kappa, mixed):
    if kappa > MAX_KAPPA.get(name, 1e4):
        pytest.skip("float32 rated unsafe at this conditioning (numerics guide)")
    build, fwd_tol, bwd_tol = LAYERS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutogradOnlyWarning)
        torch.manual_seed(0)
        layer64 = build(F64).to(F64)
        torch.manual_seed(0)
        layer32 = build(F32).to(F32)
    state = copy.deepcopy(layer64.state_dict())
    layer32.load_state_dict(state)
    if mixed:
        layer32.eigh_dtype = F64
    layer64.train()
    layer32.train()
    x32 = _spd(kappa, seed=int(kappa) % 97)
    with torch.no_grad():
        shape = layer64(x32.double()).shape
    layer64.load_state_dict(state)  # undo the running statistics of that pass
    cotangent = torch.randn(
        shape, dtype=F64, generator=torch.Generator().manual_seed(1)
    )
    out64, grad64 = _run(layer64, x32.double(), cotangent)
    out32, grad32 = _run(layer32, x32, cotangent)
    assert out32.dtype == F32
    assert torch.isfinite(out32).all() and torch.isfinite(grad32).all()
    assert _relative(out32, out64) < fwd_tol
    assert _relative(grad32, grad64) < bwd_tol

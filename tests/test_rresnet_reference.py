"""Default residual blocks and models against outputs saved before the fused block.

``fixtures/rresnet_reference.pt`` was saved on commit 4b8aa2d
(feat/numerics-manual-backward: ReEig floor, eigvalsh, Cholesky norm,
AffineInvariantExp, eigenvalue clamp: 5 eigendecompositions and a Cholesky per
affine-invariant block) with ``python tests/test_rresnet_reference.py save``.
Each case stores the constructor kwargs, the initial state dict (loaded here,
so the initialization order does not matter), the input, the cotangent, the
training-mode output, the input and parameter gradients and the evaluation
output after the training pass. With the default options, the fused block
(2 eigendecompositions) gives the same values to rounding (float64, condition
number 1e2).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from yetanotherspdnet.model import GBWBNRResNet, RResNet
from yetanotherspdnet.nn.rresnet_layers import ResidualBlock


PATH = Path(__file__).parent / "fixtures" / "rresnet_reference.pt"
F64 = torch.float64
N, N_MODEL, BATCH, SEED, KAPPA = 6, 8, 8, 1234, 1e2
CLASSES = {
    "ResidualBlock": ResidualBlock,
    "RResNet": RResNet,
    "GBWBNRResNet": GBWBNRResNet,
}
MODEL = {"input_dim": N_MODEL, "output_dim": 3}
CONFIGS = {
    "ResidualBlock[affine_invariant]": ("ResidualBlock", {"n_features": N}),
    "ResidualBlock[affine_invariant,mlp,dynamic]": (
        "ResidualBlock",
        {
            "n_features": N,
            "spectrum_type": "mlp",
            "stiefel_parametrization_mode": "dynamic",
        },
    ),
    "ResidualBlock[affine_invariant,autograd]": (
        "ResidualBlock",
        {"n_features": N, "use_autograd": True},
    ),
    "ResidualBlock[log_euclidean]": (
        "ResidualBlock",
        {"n_features": N, "metric": "log_euclidean"},
    ),
    "RResNet": (
        "RResNet",
        {**MODEL, "hidden_layers_size": [6, 4], "n_residual_blocks": [2, 1]},
    ),
    "RResNet[log_euclidean,batchnorm]": (
        "RResNet",
        {
            **MODEL,
            "hidden_layers_size": [6],
            "n_residual_blocks": [1],
            "residual_metric": "log_euclidean",
            "batchnorm": True,
        },
    ),
    "GBWBNRResNet": ("GBWBNRResNet", {**MODEL, "hidden_dim": 6}),
}


def _input(n):
    generator = torch.Generator().manual_seed(SEED)
    q, _ = torch.linalg.qr(torch.randn(BATCH, n, n, generator=generator, dtype=F64))
    u = torch.rand(BATCH, n, generator=generator, dtype=F64)
    x = (q * (KAPPA ** (u - 0.5)).unsqueeze(-2)) @ q.mT
    return (x + x.mT) / 2


def _build(cls_name, kwargs):
    torch.manual_seed(SEED)
    return CLASSES[cls_name](
        **kwargs, dtype=F64, generator=torch.Generator().manual_seed(SEED)
    )


def _run(module, data, cotangent=None):
    module.train()
    data = data.clone().requires_grad_(True)
    out = module(data)
    if cotangent is None:
        cotangent = torch.randn(
            out.shape, generator=torch.Generator().manual_seed(SEED + 1), dtype=F64
        )
    names = [name for name, p in module.named_parameters() if p.requires_grad]
    params = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out, [data, *params], cotangent, allow_unused=True)
    module.eval()
    with torch.no_grad():
        out_eval = module(data.detach())
    return {
        "cotangent": cotangent,
        "output": out.detach(),
        "input_grad": grads[0],
        "param_grads": dict(zip(names, (g for g in grads[1:]), strict=True)),
        "output_eval": out_eval,
    }


def _save():
    reference = {}
    for name, (cls_name, kwargs) in CONFIGS.items():
        module = _build(cls_name, kwargs)
        data = _input(kwargs.get("n_features", N_MODEL))
        case = {"state_dict": {k: v.clone() for k, v in module.state_dict().items()}}
        case.update(_run(module, data))
        reference[name] = case
    PATH.parent.mkdir(exist_ok=True)
    torch.save(reference, PATH)


@pytest.fixture(scope="module")
def reference():
    return torch.load(PATH, weights_only=True)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_default_outputs_unchanged(name, reference):
    cls_name, kwargs = CONFIGS[name]
    case = reference[name]
    module = _build(cls_name, kwargs)
    module.load_state_dict(case["state_dict"])
    new = _run(module, _input(kwargs.get("n_features", N_MODEL)), case["cotangent"])
    for key in ("output", "input_grad", "output_eval"):
        assert_close(new[key], case[key], rtol=1e-10, atol=1e-12)
    for param, grad in case["param_grads"].items():
        if grad is None:
            assert new["param_grads"][param] is None
        else:
            assert_close(new["param_grads"][param], grad, rtol=1e-10, atol=1e-12)


if __name__ == "__main__" and sys.argv[1:] == ["save"]:
    _save()

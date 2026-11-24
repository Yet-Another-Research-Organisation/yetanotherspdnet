# Testing

The suite lives in `tests/` and runs with `pytest` (the `test` extra). It
measures coverage by default and fails below 40 %.

```bash
pytest                                              # full suite, with coverage
pytest tests/nn/test_base.py::TestBiMap --no-cov    # one class, without the threshold
pytest -m "not slow"                                # skip the tests marked slow
python -W error::DeprecationWarning -m pytest       # fail on deprecated names
```

`pytest` always passes `--cov` (see `[tool.pytest.ini_options]`), so a partial
run needs `--no-cov` to skip the coverage threshold. A CPU with a few threads
is enough: the suite does not need a GPU, and uses the GPU when one is
available (`device` fixture).

## Layout

| path | what it checks |
|---|---|
| `tests/functions/` | matrix functions, divided differences, Stiefel projections, M-estimators; one module per geometry in `spd_geometries/` (means, geodesics, transports, the registry) |
| `tests/nn/` | layers, batch normalizations, parametrizations, residual blocks and their options |
| `tests/test_model.py` | the three models: shapes, options, hooks, hashes |
| `tests/test_manual_backward.py` | manual backward is the norm: the autograd graph of every layer and model built with the defaults holds no decomposition outside `AUTOGRAD_ONLY`; `gradcheck` of every manual `Function` |
| `tests/test_float32.py` | every layer in float32 (and mixed precision) against float64 on the same rounded input, $\kappa \le 10^4$ (`MAX_KAPPA` lists the exceptions) |
| `tests/test_numerics.py` | dtype-aware floors, guarded standard deviations |
| `tests/test_regressions.py` | one test per bug fixed after the audit; each failed before its fix |
| `tests/test_rresnet_reference.py` | the residual blocks and models against outputs saved before the fused block (`tests/fixtures/rresnet_reference.pt`) |
| `tests/test_deprecation.py` | every deprecated alias warns once and behaves like its replacement |

## Conventions

- **Fixtures** `device`, `dtype` (float64), `seed` and `generator` are shared
  in `tests/conftest.py`; a module that needs another value overrides them.
- **Both gradient paths.** An operation with a manual backward is tested with
  `torch.autograd.gradcheck` in float64, including repeated and nearly repeated
  eigenvalues, the identity and $\kappa = 10^{6}$, and against the autograd
  path:

```python
import torch
from torch.testing import assert_close

from yetanotherspdnet.functions.spd_linalg import LogmSPD, logm_SPD
from yetanotherspdnet.random.spd import random_SPD


def test_logm_gradients_agree():
    X = random_SPD(
        10, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    X_auto, X_manual = X.clone().requires_grad_(), X.clone().requires_grad_()
    logm_SPD(X_auto)[0].sum().backward()
    LogmSPD.apply(X_manual).sum().backward()
    assert_close(X_auto.grad, X_manual.grad)
    # the manual backwards assume symmetric inputs: symmetrize the perturbations
    assert torch.autograd.gradcheck(
        lambda A: LogmSPD.apply((A + A.mT) / 2), (X[:2].clone().requires_grad_(),)
    )


test_logm_gradients_agree()
```

- **Exactness claims are tested**: the Bures–Wasserstein transport against an
  integration of the transport equation, the weighted barycenter against the
  geodesic, the GBWBN layer against the formulas of its paper
  (`TestGBWBNConformity`), the residual step against its unit length.
- **Regressions** go in `tests/test_regressions.py`, with a docstring saying
  what failed.

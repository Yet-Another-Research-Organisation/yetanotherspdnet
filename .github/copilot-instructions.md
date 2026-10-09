# Copilot Instructions

## Project Overview

**yetanotherspdnet** is a PyTorch library for neural networks that operate directly on Symmetric Positive Definite (SPD) matrices using Riemannian geometry. It provides SPD matrix operations, multiple geometric means, and neural network layers (BiMap, ReEig, LogEig, Vec/Vech, BatchNormSPDMean).

## Build, Test & Lint

```bash
# Install (use [all] to include test, dev, and docs extras)
pip install -e ".[all]"

# Run all tests
pytest

# Run a single test file
pytest tests/nn/test_base.py

# Run a single test function
pytest tests/test_model.py::TestSPDnet::test_initialization

# Skip slow tests
pytest -m "not slow"

# Format and lint
ruff format .
ruff check --fix .

# Type check
mypy src/

# Build docs
cd docs && make html
```

**Pytest markers:** `slow`, `integration`  
**Coverage minimum:** 40% (enforced in CI)

## Architecture

```
src/yetanotherspdnet/
├── model.py                    # SPDnet: the high-level model class
├── functions/
│   ├── spd_linalg.py           # Core SPD matrix ops: logm, sqrtm, powm, vec, vech, …
│   ├── scalar_functions.py     # Scalar helpers: sqrt_deriv, inv_sqrt, softplus, …
│   ├── stiefel.py              # Stiefel manifold projections (polar, QR)
│   └── spd_geometries/         # One file per geometry (affine_invariant, log_euclidean,
│                               # kullback_leibler, kullback_leibler_symmetrized)
├── nn/
│   ├── base.py                 # Layers: BiMap, ReEig, LogEig, Vec, Vech
│   ├── batchnorm.py            # BatchNormSPDMean (uses spd_geometries for the mean)
│   └── parametrizations.py     # nn.utils.parametrize wrappers for orthogonality constraints
└── random/
    ├── spd.py                  # random_SPD(), random_DPD()
    └── stiefel.py              # _init_weights_stiefel()
```

**Dependency flow:**
- `model.py` → `nn/` layers + `functions/`
- `nn/base.py` → `functions/spd_linalg.py`
- `nn/batchnorm.py` → `functions/spd_geometries/`
- `nn/parametrizations.py` → enforces Stiefel constraints on BiMap weights

**Tests mirror source structure:** `tests/functions/`, `tests/functions/spd_geometries/`, `tests/nn/`, `tests/test_model.py`.

## Key Conventions

**PyTorch patterns:**
- All layers extend `nn.Module`; custom differentiable ops use `torch.autograd.Function`
- Orthogonality constraints on BiMap weights are enforced via `torch.nn.utils.parametrize`
- All ops must support arbitrary `device` and `dtype` (float32/float64, CPU/CUDA)

**Testing patterns:**
- Fixtures `device`, `dtype`, `generator` are defined in `tests/conftest.py`; use them for reproducibility and device-agnostic tests
- Parametrize over `(device, dtype)` combinations when testing numeric ops
- Gradient correctness is verified with numerical Jacobian checks (`torch.autograd.gradcheck`)
- Slow or integration tests must be tagged with `@pytest.mark.slow` / `@pytest.mark.integration`

**Geometry convention:**
- Each file under `spd_geometries/` implements a self-contained geometry (geodesic, mean, distance, …)
- `batchnorm.py` selects the geometry at construction time; adding a new geometry means adding a file there and wiring it into `batchnorm.py`

**Python version:** 3.11+ (use `list[int]` etc., not `List[int]`)  
**Line length:** 88 characters (ruff default)  
**Public API:** exposed through `__all__` in each `__init__.py`; top-level exports are `functions`, `nn`, `random`, `SPDnet`

## Code Quality Standards

- Write clear, compact, and human-readable code without emojis
- Avoid code duplication; reuse functions and modules whenever possible
- Prioritize execution speed, memory efficiency, and CPU/GPU optimization
- Make all modifications using the minimum amount of code possible
- Ensure all changes are easily explainable and understandable by humans
- Include comments for complex logic or non-obvious optimizations
- Favor readability over clever or overly concise implementations

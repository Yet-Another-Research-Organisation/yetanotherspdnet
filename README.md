# Yet Another SPDNet

[![Tests](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/tests.yml/badge.svg)](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Yet-Another-Research-Organisation/yetanotherspdnet/branch/main/graph/badge.svg)](https://codecov.io/gh/Yet-Another-Research-Organisation/yetanotherspdnet)
[![Documentation](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/docs.yml/badge.svg)](https://yet-another-research-organisation.github.io/yetanotherspdnet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A robust and tested implementation of SPDNet learning models for **Symmetric Positive Definite (SPD) matrices** using Riemannian geometry.

## Overview

SPDNet is a neural network architecture that operates directly on SPD matrices. These matrices arise naturally in many domains: covariance estimation, diffusion tensor imaging, radar signal processing, and brain-computer interfaces.

The library provides:

- **Core SPD matrix operations**: matrix logarithm, square root, inverse square root, matrix power, congruence transforms, whitening
- **Riemannian geometry**: affine-invariant, log-Euclidean, and symmetrized Kullback-Leibler geometries
- **Neural network layers**: BiMap (projection), ReEig (eigenvalue rectification), LogEig (tangent space), BatchNorm for SPD matrices
- **Learnable parametrizations**: SPD and Stiefel manifold constraints for weight matrices
- **Manual gradients**: custom `torch.autograd.Function` implementations for numerical stability in float64

All operations run in float64 on GPU by default for numerical stability (eigendecompositions).

## Installation

### From Source

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[all]"
```

## Quick Start

```python
import torch
from yetanotherspdnet.model import SPDnet
from yetanotherspdnet.random.spd import random_SPD

# Generate random SPD matrices (batch of 32, size 50x50)
X = random_SPD(n_features=50, n_matrices=32, cond=100, device="cuda", dtype=torch.float64)

# Create an SPDNet model
model = SPDnet(
    input_dim=50,
    hidden_layers_size=[30, 20],
    output_dim=10,
    softmax=True,
    batchnorm=True,
    device="cuda",
    dtype=torch.float64,
)

# Forward pass
output = model(X)
print(output.shape)  # (32, 10)
```

### SPD Matrix Operations

```python
from yetanotherspdnet.functions.spd_linalg import (
    logm_SPD,
    sqrtm_SPD,
    inv_sqrtm_SPD,
    powm_SPD,
)
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_geodesic,
    AffineInvariantMean,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import LogEuclideanMean

# Matrix functions (return (result, eigvals, eigvecs) tuples)
X_log = logm_SPD(X)[0]           # Matrix logarithm
X_sqrt = sqrtm_SPD(X)[0]         # Matrix square root
X_sqrt_inv = inv_sqrtm_SPD(X)[0] # Inverse square root

# Riemannian means (manual gradient, GPU-efficient)
mean_ai = AffineInvariantMean(X)  # Affine-invariant (Karcher) mean
mean_le = LogEuclideanMean(X)     # Log-Euclidean mean

# Geodesic between two SPD matrices at parameter t in [0, 1]
G = affine_invariant_geodesic(X[0], X[1], t=0.5)
```

### Neural Network Layers

```python
from yetanotherspdnet.nn import BiMap, ReEig, LogEig, BatchNormSPDMean

bimap = BiMap(input_size=50, output_size=30, device="cuda", dtype=torch.float64)
reeig = ReEig(eps=1e-4)
logeig = LogEig()
bn = BatchNormSPDMean(n_features=30, device="cuda", dtype=torch.float64)
```

## Documentation

[Full Documentation](https://yet-another-research-organisation.github.io/yetanotherspdnet/)

- [Installation Guide](https://yet-another-research-organisation.github.io/yetanotherspdnet/installation.html)
- [Quick Start Tutorial](https://yet-another-research-organisation.github.io/yetanotherspdnet/quickstart.html)
- [API Reference](https://yet-another-research-organisation.github.io/yetanotherspdnet/api.html)
- [Contributing Guide](https://yet-another-research-organisation.github.io/yetanotherspdnet/contributing.html)

## Development

### Setup

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e ".[all]"
pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=yetanotherspdnet --cov-report=html

# Run a specific test file
uv run pytest tests/functions/test_spd_linalg.py
```

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint
uv run ruff check src/

# Auto-fix lint issues
uv run ruff check --fix src/
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Pull request checklist:

- Tests pass (`uv run pytest`)
- Coverage >= 40% (`--cov-fail-under=40`)
- Code formatted (`uv run ruff format src/`)
- No linting errors (`uv run ruff check src/`)
- Documentation updated if needed
- Type hints on public functions

## Project Structure

```
yetanotherspdnet/
├── src/yetanotherspdnet/
│   ├── functions/
│   │   ├── spd_linalg.py              # Core SPD linear algebra
│   │   ├── scalar_functions.py        # Element-wise functions on eigenvalues
│   │   └── spd_geometries/
│   │       ├── affine_invariant.py    # Affine-invariant geometry
│   │       ├── log_euclidean.py       # Log-Euclidean geometry
│   │       ├── kullback_leibler.py    # Base geometries (arithmetic, harmonic)
│   │       └── kullback_leibler_symmetrized.py  # KL-sym + adaptive geodesic
│   ├── nn/
│   │   ├── base.py                    # BiMap, ReEig, LogEig, Vec, Vech layers
│   │   ├── batchnorm.py               # SPD batch normalization
│   │   └── parametrizations.py        # SPD/Stiefel parametrizations
│   ├── random/
│   │   ├── spd.py                     # Random SPD matrix generation
│   │   └── stiefel.py                 # Random Stiefel matrix generation
│   └── model.py                       # SPDnet model definition
├── tests/                             # Test suite (3789 tests)
├── docs/                              # Sphinx documentation
├── pyproject.toml                     # Project configuration
└── README.md
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- SciPy >= 1.11.0 (for tests only)

## Authors

- [Ammar Mian](https://github.com/ammarmian)
- Florent Bouchard
- Guillaume Ginolhac
- Matthieu Gallet

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Documentation](https://yet-another-research-organisation.github.io/yetanotherspdnet/)
- [Issue Tracker](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues)
- [Discussions](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/discussions)
- [Releases](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/releases)

# Yet Another SPDNet

[![Tests](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/tests.yml/badge.svg)](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Yet-Another-Research-Organisation/yetanotherspdnet/branch/main/graph/badge.svg)](https://codecov.io/gh/Yet-Another-Research-Organisation/yetanotherspdnet)
[![Documentation](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/actions/workflows/docs.yml/badge.svg)](https://yet-another-research-organisation.github.io/yetanotherspdnet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A robust and tested implementation of SPDNet learning models for **Symmetric Positive Definite (SPD) matrices**.

## Overview

SPDNet is a neural network architecture designed to work directly with SPD matrices, which naturally arise in many domains.
TODO: Add description


## Features

TODO

## Installation

### From Source

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[all]"  # Install all dependencies (test, dev, docs)
```

## Quick Start

```python
import torch
from yetanotherspdnet import spd
from yetanotherspdnet.models import SPDNet

# Generate random SPD matrices
X = spd.random_SPD(n_features=50, n_matrices=32, cond=100)

# Create an SPDNet model
model = SPDNet(
    input_dim=50,
    hidden_layers_size=[30, 20],
    output_dim=10,
    softmax=True,
    batchnorm=True,
)

# Forward pass
output = model(X)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### SPD Matrix Operations

```python
# Matrix operations
X_log = spd.logm_SPD(X)              # Matrix logarithm
X_sqrt = spd.sqrtm_SPD(X)            # Matrix square root
X_inv_sqrt = spd.inv_sqrtm_SPD(X)    # Inverse square root

# Riemannian means
from yetanotherspdnet import means

mean_geom = means.mean_geometric(X)         # Geometric (Karcher) mean
mean_log = means.mean_log_euclidean(X)      # Log-Euclidean mean
```

## Documentation

📚 **[Full Documentation](https://yet-another-research-organisation.github.io/yetanotherspdnet/)**

- [Installation Guide](https://yet-another-research-organisation.github.io/yetanotherspdnet/installation.html)
- [Quick Start Tutorial](https://yet-another-research-organisation.github.io/yetanotherspdnet/quickstart.html)
- [API Reference](https://yet-another-research-organisation.github.io/yetanotherspdnet/api.html)
- [Contributing Guide](https://yet-another-research-organisation.github.io/yetanotherspdnet/contributing.html)

## Development

### Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=yetanotherspdnet --cov-report=html

# Run specific test file
pytest tests/test_spd.py
```

### Code Quality

```bash
# Format code with ruff
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking (optional)
mypy src/
```

### Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick checklist for PRs:**
- ✅ Tests pass (`pytest`)
- ✅ Coverage ≥80%
- ✅ Code formatted (`ruff format .`)
- ✅ No linting errors (`ruff check .`)
- ✅ Documentation updated
- ✅ Type hints added

## Project Structure

```
yetanotherspdnet/
├── src/yetanotherspdnet/     # Main package (src-layout)
│   ├── spd.py                # SPD matrix operations
│   ├── nn.py                 # Neural network layers
│   ├── means.py              # SPD mean computations
│   ├── stiefel.py            # Stiefel manifold utilities
│   └── models.py             # High-level models
├── tests/                    # Test suite
├── docs/                     # Sphinx documentation
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Requirements

- Python ≥3.11
- PyTorch ≥2.0.0
- SciPy ≥1.11.0 (for tests only)

## Authors

- [Ammar Mian](https://github.com/ammarmian)
- Florent Bouchard
- Guillaume Ginolhac
- Matthieu Gallet

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- 📖 [Documentation](https://yet-another-research-organisation.github.io/yetanotherspdnet/)
- 🐛 [Issue Tracker](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues)
- 💬 [Discussions](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/discussions)
- 📦 [Releases](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/releases)

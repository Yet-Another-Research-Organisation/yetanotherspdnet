# Installation

## Prerequisites

- Python 3.11 or higher
- PyTorch 2.0.0 or higher

## Install from Source

To install the latest development version from source:

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e .
```

## Install with Development Dependencies

To install with all development tools (testing, linting, docs):

```bash
pip install -e ".[all]"
```

Or install specific groups:

```bash
# For testing
pip install -e ".[test]"

# For development
pip install -e ".[dev]"

# For building documentation
pip install -e ".[docs]"
```

## Verify Installation

You can verify the installation by running:

```python
import yetanotherspdnet
print(yetanotherspdnet.__version__)

# Try importing main modules
from yetanotherspdnet import spd, nn, means
from yetanotherspdnet.models import SPDNet
```

## Running Tests

If you installed with test dependencies, you can run the test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=yetanotherspdnet --cov-report=html
```

## Common Issues

### PyTorch Installation

If you encounter issues with PyTorch, install it separately first:

```bash
# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA (check PyTorch website for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### SciPy (for tests only)

SciPy is only required for running tests. If you're not running tests, you don't need it.

```bash
pip install scipy
```

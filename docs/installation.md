# Installation

## Requirements

- Python 3.11 or later
- PyTorch 2.0 or later (the only runtime dependency)

## From GitHub

```bash
pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
```

A branch or a tag can be pinned with `...yetanotherspdnet.git@<ref>`.

## From source, for development

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e ".[all]"   # test + dev + docs extras
```

The extras can also be installed separately:

| Extra | Contents |
|---|---|
| `test` | pytest, pytest-cov, SciPy (reference implementations used by the tests) |
| `dev` | ruff (pinned, same version as CI), mypy, pre-commit |
| `docs` | Sphinx, MyST, Furo theme |

## Check the installation

```python
import torch
import yetanotherspdnet
from yetanotherspdnet import SPDnet
from yetanotherspdnet.random.spd import random_SPD

print(yetanotherspdnet.__version__)
X = random_SPD(n_features=8, n_matrices=4, dtype=torch.float64)
print(SPDnet(input_dim=8, hidden_layers_size=[4], output_dim=2)(X).shape)
# torch.Size([4, 2])
```

## Run the tests

```bash
pytest                                   # full suite, coverage enforced (>= 40%)
pytest tests/nn/test_base.py::TestBiMap --no-cov  # a single class
```

## PyTorch builds

If the default PyTorch wheel does not suit your machine, install PyTorch first
from the [official selector](https://pytorch.org/get-started/locally/), then
install this package.

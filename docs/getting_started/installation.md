# Installation

The library needs Python 3.11 or later and has a single runtime dependency,
PyTorch 2.0 or later.

| I want to… | Command | Section |
|---|---|---|
| use the library | `pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"` | [Use](#use-the-library) |
| run it on a GPU | install the CUDA build of PyTorch first | [GPU](#cpu-or-gpu-pytorch) |
| modify it, run the tests | `pip install -e ".[all]"` in a clone | [Extras](#extras) |
| build this documentation | `pip install -e ".[docs]"`, then `make -C docs html` | {doc}`../development/documentation` |

## Use the library

```bash
pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
# or with uv
uv pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
```

`main` is the latest released state. Pin a tag or a commit for reproducible
experiments, `...yetanotherspdnet.git@<tag-or-sha>`; in a `pyproject.toml`:

```toml
dependencies = [
  "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git@main",
]
```

Check that everything is importable and runs:

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

```{note}
Layers and models default to float64. `random_SPD` follows PyTorch's default
dtype (float32) unless told otherwise, hence `dtype=torch.float64` above.
float32 and mixed precision are possible and measured in
{doc}`../numerics/precision`.
```

## CPU or GPU PyTorch

`pip` installs the PyTorch build of the index it uses, often CPU-only. For a
GPU, install PyTorch first from the
[official selector](https://pytorch.org/get-started/locally/), matching your
driver, then this package. For example, with a CUDA 13.0 driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
python -c "import torch; print(torch.cuda.is_available())"   # True
```

For a CPU-only machine, the CPU wheels are smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Every layer and function takes `device` and `dtype`, so a model moves to the
GPU like any PyTorch model. Two things to know before choosing:

- for matrices of a few hundred rows, a CPU run with a few threads is often as
  fast as a small GPU (the work is dominated by batched eigendecompositions);
- LAPACK's eigendecomposition (CPU) is more robust than cuSOLVER's (GPU) on
  nearly singular batches ({doc}`../numerics/stability`).

## Extras

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e ".[all]"      # test + dev + docs extras
pre-commit install           # ruff on every commit, same version as the CI
```

| Extra | Contents |
|---|---|
| `test` | pytest, pytest-cov, SciPy (reference implementations used by the tests) |
| `dev` | ruff (pinned, same version as the CI), mypy, pre-commit |
| `docs` | Sphinx, MyST, pydata-sphinx-theme, sphinx-design, sphinx-copybutton, and numpy + matplotlib for the diagrams |
| `all` | the three above |

Running the tests and building the documentation are described in
{doc}`../development/testing` and {doc}`../development/documentation`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double` | float32 input to a float64 model: create the data with `dtype=torch.float64`, or build the model with `dtype=torch.float32` (and maybe `eigh_dtype=torch.float64`). |
| `linalg.eigh: The algorithm failed to converge` | Nearly singular or very ill-conditioned matrices, mostly on GPU or in float32: run the decompositions on CPU or in float64 (`eigh_dtype`), check the conditioning of the data; for GBWBN see {doc}`../user_guide/batchnorm`. |
| `AutogradOnlyWarning` | An operation without a manual backward is differentiated by autograd (for instance the static `BiMap` parametrization); the list and the alternatives are in {doc}`../numerics/manual_backward`. |
| `DeprecationWarning: … is deprecated, use … instead` | A renamed function; see {doc}`../reference/deprecated`. |
| `pytest: error: unrecognized arguments: --cov=...` | The `test` extra is missing: `pip install -e ".[test]"`. |

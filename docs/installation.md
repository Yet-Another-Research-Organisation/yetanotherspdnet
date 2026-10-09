# Installation

The library needs Python 3.11 or later and has a single runtime dependency,
PyTorch 2.0 or later. Pick the line that matches what you want to do:

| I want to… | Command | Section |
|---|---|---|
| use the library | `pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"` | [Use](#use-the-library) |
| run it on a GPU | install the CUDA build of PyTorch first | [GPU](#gpu) |
| modify it, run the tests | `pip install -e ".[all]"` in a clone | [Develop](#develop) |
| build this documentation | `pip install -e ".[docs]"`, then `make -C docs html` | [Documentation](#build-the-documentation) |

## Use the library

```bash
pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
# or with uv
uv pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
```

`main` is the latest released state. Pin a tag or a commit for reproducible
experiments: `...yetanotherspdnet.git@<tag-or-sha>`. In a `pyproject.toml`:

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
Work in float64 (the default of every layer): eigendecompositions of
ill-conditioned matrices lose accuracy quickly in float32. `random_SPD`
returns float32 unless told otherwise, hence `dtype=torch.float64` above.
```

## GPU

`pip` installs the PyTorch build of the index it uses, often CPU-only. For a
GPU, install PyTorch first from the
[official selector](https://pytorch.org/get-started/locally/), matching your
driver, then this package. For example, with a CUDA 13.0 driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git"
python -c "import torch; print(torch.cuda.is_available())"   # True
```

Every layer and function takes `device` and `dtype`, so a model moves to the
GPU like any PyTorch model. For matrices of a few hundred rows, a CPU run with
a few threads is often as fast as a small GPU, and LAPACK's eigendecomposition
is more robust than cuSOLVER's on nearly singular batches
({doc}`user_guide/numerics`).

## Develop

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
| `docs` | Sphinx, MyST, Furo theme, sphinx-design, sphinx-copybutton |

Run the tests:

```bash
pytest                                              # full suite, coverage enforced (>= 40 %)
pytest tests/nn/test_base.py::TestBiMap --no-cov    # one class, without coverage
ruff format . && ruff check .                       # what the CI lint job runs
```

`pytest` always passes `--cov` (see `pyproject.toml`), so it needs the `test`
extra; `--no-cov` skips the coverage threshold for a partial run.

## Build the documentation

```bash
pip install -e ".[docs]"
make -C docs html              # docs/_build/html/index.html
python docs/_diagrams/make_diagrams.py   # only after changing a diagram
```

The online documentation is built from `main` by the `Documentation`
workflow, at each release or on demand (`gh workflow run docs.yml --ref main`).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: cannot import name 'functions' from partially initialized module` | An old version installed only the top-level package; reinstall from `main`. |
| `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double` | float32 input to a float64 model: create the data with `dtype=torch.float64` (or build the model with `dtype=torch.float32`). |
| `linalg.eigh: The algorithm failed to converge` on GPU | Nearly singular matrices; run on CPU or check the conditioning of the data (for GBWBN, use `bw_theta < 1`). |
| `pytest: error: unrecognized arguments: --cov=...` | The `test` extra is missing: `pip install -e ".[test]"`. |

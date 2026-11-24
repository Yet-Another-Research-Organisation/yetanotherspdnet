# Contributing

This page is the contributing guide of the project (the repository's
`CONTRIBUTING.md` points here).

## Setup

```bash
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
cd yetanotherspdnet
pip install -e ".[all]"      # test + dev + docs extras
pre-commit install           # ruff (lint + format) on every commit
```

Fork the repository first if you do not have write access, and clone your
fork instead.

## Workflow

- **Branches**: `main` is protected and only receives pull requests. Name
  branches by their purpose: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`,
  `refactor/<topic>`.
- **Commits**: conventional commits, one focused change per commit:
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`, with an
  optional scope (`fix(gbwbn): ...`).
- **Pull requests** target `main`; describe what changes, why, and how it was
  tested (and, for numerical changes, the before / after numbers). Keep
  history: merge, do not force-push shared branches.

## Code

- **Style**: `ruff format .` and `ruff check .`, with the ruff version pinned
  in three places that must stay in sync: the `dev` extra of `pyproject.toml`,
  `.github/workflows/tests.yml` and `.pre-commit-config.yaml`. The rule sets
  (including pydocstyle with the NumPy convention, pep8-naming and pylint
  refactor rules) are in `[tool.ruff]` of `pyproject.toml`.
- **Types**: annotate public functions; `mypy src` runs in the CI (options in
  `[tool.mypy]`).
- **Inputs**: validate public arguments with `ValueError` / `TypeError`, not
  `assert`.
- **Constants**: numerical thresholds live in `functions/_constants.py`.
- **Renames**: never break a public name; keep the old one as a deprecated
  alias with `yetanotherspdnet._deprecation._deprecated` (it appears in
  {doc}`../reference/deprecated`).

### Numerical code

- A new operation built on a matrix decomposition comes in two versions: a
  `snake_case` function differentiated by autograd and a `CamelCase`
  `torch.autograd.Function` (or a function chaining such classes) with a
  hand-written backward. Layers select one with `functions.dispatch.select_path`
  or a `DualPath`.
- Manual backward is the norm: if an operation has no manual backward, add it
  to `functions.dispatch.AUTOGRAD_ONLY` with the reason and call
  `warn_autograd_only` where a layer reaches it; the test suite fails on any
  unlisted decomposition in a default autograd graph
  ({doc}`../numerics/manual_backward`).
- Use the closed-form divided differences of `spd_linalg` for new spectral
  functions, and the dtype-aware floors (`spd_floor`) instead of absolute
  thresholds.

### Docstrings

NumPy style. Classes get a class docstring (description, `Attributes`);
constructor arguments are documented in the `__init__` docstring; the API
reference shows both. Use raw strings (`r"""`) for docstrings with LaTeX.

```python
import torch


def my_function(X: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    r"""
    Short one-line description.

    Longer description, with the formula :math:`f(X) = U f(\Lambda) U^\top`.

    Parameters
    ----------
    X : torch.Tensor of shape (..., n, n)
        SPD matrices
    eps : float, optional
        Regularization parameter. Default is 1e-3

    Returns
    -------
    torch.Tensor of shape (..., n, n)
        Output matrices
    """
    return X
```

## Pull request checklist

- [ ] `pytest` passes (see {doc}`testing`), with tests for the new code on
  both gradient paths
- [ ] `ruff format --check .` and `ruff check .` pass
- [ ] `mypy src` passes
- [ ] the documentation builds with `-W` (see {doc}`documentation`), and the
  user guide is updated if behaviour or options change
- [ ] public names unchanged, or deprecated aliases added

## What the CI checks

| job | command | when |
|---|---|---|
| tests | `pytest` (coverage enforced, `--cov-fail-under=40`) on Python 3.11, 3.12 and 3.13 | every push and pull request |
| lint | `ruff check .`, `ruff format --check .` | every push and pull request |
| docs | `sphinx-build -W --keep-going -b html docs docs/_build/html` | every push and pull request |
| type check | `mypy src` | every push and pull request |
| documentation site | build and deploy to GitHub Pages | releases, or on demand |

## Getting help

- [Issues](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues):
  bugs and feature requests.
- Be respectful and constructive in reviews and discussions.

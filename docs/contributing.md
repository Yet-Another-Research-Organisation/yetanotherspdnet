# Contributing Guide

Thank you for your interest in contributing to Yet Another SPDNet! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/YetAnotherSPDNet.git
cd YetAnotherSPDNet
```

### 2. Install Development Dependencies

```bash
pip install -e ".[all]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting checks before each commit.

## Development Workflow

### Branch Strategy

We use a simple branch-based workflow:

- **`main`**: Production-ready code (protected, requires CI to pass)
- **`feature/feature-name`**: New features
- **`bugfix/issue-description`**: Bug fixes
- **`hotfix/critical-issue`**: Critical fixes for production

### Creating a New Branch

```bash
# For new features
git checkout -b feature/my-awesome-feature

# For bug fixes
git checkout -b bugfix/fix-issue-123

# For hotfixes
git checkout -b hotfix/critical-bug
```

## Making Changes

### 1. Code Style

We use **ruff** for linting and formatting (replaces black, isort, flake8):

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

### 2. Type Checking

We use **mypy** for static type checking:

```bash
mypy src/
```

Add type hints to all new functions:

```python
def my_function(x: torch.Tensor, n: int = 10) -> torch.Tensor:
    """Function with type hints."""
    ...
```

### 3. Writing Tests

All new features must include tests. We use **pytest**:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=yetanotherspdnet

# Run specific test file (coverage is enforced by default, skip it here)
pytest tests/functions/test_spd_linalg.py --no-cov

# Run specific test class
pytest tests/nn/test_base.py::TestBiMap --no-cov
```

#### Test Structure

Tests use pytest fixtures for `device`, `dtype` and a seeded `generator`
(see the top of each test module). Operations that exist on both gradient
paths must be tested on both (`torch.autograd.gradcheck` in float64, plus
agreement between the autograd and manual gradients):

```python
import pytest
import torch
from torch.testing import assert_close

from yetanotherspdnet.functions.spd_linalg import LogmSPD, logm_SPD
from yetanotherspdnet.random.spd import random_SPD


@pytest.mark.parametrize("n_features, n_matrices", [(10, 5)])
def test_logm_gradients_agree(n_features, n_matrices, device, dtype, generator):
    X = random_SPD(
        n_features, n_matrices, device=device, dtype=dtype, generator=generator
    )
    X_auto = X.clone().requires_grad_()
    X_manual = X.clone().requires_grad_()
    logm_SPD(X_auto)[0].sum().backward()
    LogmSPD.apply(X_manual).sum().backward()
    assert_close(X_auto.grad, X_manual.grad)
```

### 4. Writing Documentation

We use **Sphinx** with **MyST** (Markdown support):

#### Docstring Format

Use NumPy style docstrings. Classes get a class docstring (description and an
`Attributes` section for parameters and buffers). Constructor arguments are
documented in the `__init__` docstring. The API reference shows both.

```python
def my_function(X: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Short one-line description.

    Longer description with more details about what the function does,
    including mathematical background if relevant.

    Parameters
    ----------
    X : torch.Tensor
        Input SPD matrices of shape (..., n, n)
    eps : float, optional
        Regularization parameter, by default 1e-3

    Returns
    -------
    torch.Tensor
        Output tensor of shape (..., n, n)

    Raises
    ------
    ValueError
        If X is not positive definite

    Examples
    --------
    >>> X = random_SPD(50, 10, dtype=torch.float64)
    >>> result = my_function(X, eps=1e-4)
    """
    ...
```

#### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html SPHINXOPTS="-W --keep-going"   # warnings are errors
xdg-open _build/html/index.html
```

The API reference (`docs/reference/`) is generated at build time by
sphinx-autoapi and is not committed.

## Pull Request Process

### 1. Commit Your Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new SPD mean computation method"
git commit -m "fix: correct gradient computation in LogEig"
git commit -m "docs: update installation instructions"
git commit -m "test: add tests for BiMap layer"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 2. Push to Your Fork

```bash
git push origin feature/my-awesome-feature
```

### 3. Create Pull Request

Go to GitHub and create a pull request from your fork to the main repository.

**Pull Request Checklist:**

- [ ] Code follows style guidelines (ruff passes)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new features
- [ ] Coverage not decreased (CI enforces ≥ 40%; aim for ≥ 80% on new code)
- [ ] Documentation updated (docstrings and/or markdown files)
- [ ] Type hints added for new functions
- [ ] Commit messages follow conventional commits
- [ ] No merge conflicts with main branch

### 4. Code Review

A maintainer will review your PR and may request changes. Address feedback by:

```bash
# Make changes
git add .
git commit -m "fix: address review comments"
git push origin feature/my-awesome-feature
```

### 5. Merge

Once approved and CI passes, a maintainer will merge your PR into main.

## CI/CD Pipeline

All pull requests must pass CI checks:

1. **Tests**: all tests must pass on Python 3.11 and 3.12
2. **Coverage**: the suite fails below 40% (`--cov-fail-under` in `pyproject.toml`)
3. **Linting**: `ruff check .` and `ruff format --check .` must pass, with the
   ruff version pinned in `pyproject.toml` (`dev` extra), in the CI workflow and in
   `.pre-commit-config.yaml`. Bump all three together.
4. **Type checking**: `mypy src/` is not run in CI (known pre-existing errors)

Only merges to **main** require all CI checks to pass. Development branches can be pushed without restrictions for fast iteration.

## Testing Requirements

### Coverage

`pytest` measures coverage by default and fails below 40%. New code should
come with tests covering both gradient paths.

### Markers

`slow` and `integration` markers are declared in `pyproject.toml`; deselect
with `pytest -m "not slow"`.

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or discuss ideas
- **Email**: Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

Thank you for contributing to Yet Another SPDNet!

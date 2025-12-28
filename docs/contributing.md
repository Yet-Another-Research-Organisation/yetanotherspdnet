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

# Run specific test file
pytest tests/test_spd.py

# Run specific test
pytest tests/test_spd.py::TestLogmSPDExpmSymmetric::test_LogmSPD
```

#### Test Structure

```python
from unittest import TestCase
import torch
from torch.testing import assert_close
from yetanotherspdnet import spd

class TestMyFeature(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.X = spd.random_SPD(50, 10)

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = spd.my_new_function(self.X)
        assert result.shape == self.X.shape
        assert_close(result, expected_result)
```

### 4. Writing Documentation

We use **Sphinx** with **MyST** (Markdown support):

#### Docstring Format

Use Google or NumPy style docstrings:

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
    >>> X = spd.random_SPD(50, 10)
    >>> result = my_function(X, eps=1e-4)
    """
    ...
```

#### Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

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
- [ ] Test coverage maintained (≥80%)
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

1. **Tests**: All tests must pass on Python 3.11 and 3.12
2. **Coverage**: Test coverage must be ≥80%
3. **Linting**: Ruff checks must pass
4. **Type Checking**: MyPy checks should pass (warnings OK)

Only merges to **main** require all CI checks to pass. Development branches can be pushed without restrictions for fast iteration.

## Testing Requirements

### Coverage Threshold

Maintain ≥80% test coverage:

```bash
pytest --cov=yetanotherspdnet --cov-fail-under=80
```

### Test Categories

Mark tests appropriately:

```python
import pytest

@pytest.mark.slow
def test_expensive_computation():
    """This test takes a long time."""
    ...

# Run only fast tests
pytest -m "not slow"
```

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or discuss ideas
- **Email**: Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

Thank you for contributing to Yet Another SPDNet! 🎉

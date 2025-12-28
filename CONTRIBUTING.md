# Contributing to Yet Another SPDNet

Thank you for your interest in contributing! 🎉

## Quick Start

1. Fork and clone the repository
2. Install development dependencies: `pip install -e ".[all]"`
3. Install pre-commit hooks: `pre-commit install`
4. Create a feature branch: `git checkout -b feature/your-feature`
5. Make your changes and write tests
6. Run tests: `pytest --cov=yetanotherspdnet`
7. Commit with conventional commits: `git commit -m "feat: add new feature"`
8. Push and create a pull request

## Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `hotfix/critical-issue` - Critical production fixes

## Requirements for Pull Requests

- ✅ All tests pass (`pytest`)
- ✅ Test coverage ≥80%
- ✅ Code formatted with ruff (`ruff format .`)
- ✅ No linting errors (`ruff check .`)
- ✅ Documentation updated (docstrings and markdown files)
- ✅ Type hints added for new functions
- ✅ Conventional commit messages

## CI/CD

Only merges to **main** branch require CI tests to pass. This ensures:
- Fast iteration on development branches
- Production-quality code on main
- All tests pass before merge
- Coverage threshold maintained

Development branches can be pushed freely for collaboration.

## Commit Message Format

Use conventional commits:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

Examples:
```bash
git commit -m "feat: add geometric mean computation"
git commit -m "fix: correct gradient in LogEig layer"
git commit -m "docs: update installation guide"
```

## Code Style

We use **ruff** for formatting and linting:

```bash
# Format code
ruff format .

# Check and fix linting issues
ruff check --fix .
```

## Testing

Write tests for all new features:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=yetanotherspdnet --cov-report=html

# Run specific test file
pytest tests/test_spd.py
```

## Documentation

- Add docstrings to all public functions (Google/NumPy style)
- Update relevant markdown files in `docs/`
- Build docs locally: `cd docs && make html`

## Getting Help

- 📝 [Full Contributing Guide](https://yet-another-research-organisation.github.io/yetanotherspdnet/contributing.html)
- 🐛 [Report Issues](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues)
- 💬 [GitHub Discussions](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/discussions)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the project
- Show empathy towards other contributors

Thank you for contributing! 🙏

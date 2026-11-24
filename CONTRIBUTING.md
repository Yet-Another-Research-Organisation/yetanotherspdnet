# Contributing to Yet Another SPDNet

The full guide (workflow, code style, numerical conventions, docstrings,
pull request checklist, what the CI checks) is in the documentation:
[Development › Contributing](https://yet-another-research-organisation.github.io/yetanotherspdnet/development/contributing.html)
(source: `docs/development/contributing.md`).

In short:

```bash
pip install -e ".[all]" && pre-commit install       # setup
pytest                                              # tests (coverage enforced)
ruff format . && ruff check . && mypy src           # what the CI lint jobs run
```

Report bugs and request features in the
[issues](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues).

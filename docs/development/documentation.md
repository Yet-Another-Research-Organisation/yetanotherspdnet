# Documentation

These pages are built with Sphinx, MyST Markdown and the pydata-sphinx-theme
(the design of the spdnet-datasets documentation: dark mode by default, one
accent colour per section in `_static/custom.css`).

```bash
pip install -e ".[docs]"
make -C docs html SPHINXOPTS="-W --keep-going"   # warnings are errors, as in the CI
xdg-open docs/_build/html/index.html
```

## Layout

| path | contents |
|---|---|
| `docs/index.md` | landing page (hero, tiles) |
| `docs/getting_started/` | installation, quickstart |
| `docs/user_guide/` | one page per topic, each in three parts: mathematics, implementation, options and strategy |
| `docs/numerics/` | manual backward, precision, stability |
| `docs/reference/` | API reference, written by hand around `autoclass` / `automodule` (inside `{eval-rst}` blocks) |
| `docs/development/` | these pages |
| `docs/references.md` | bibliography, with DOIs and arXiv identifiers; cite with `` {ref}`[Author Year] <ref-key>` `` |
| `docs/_ext/` | the local Sphinx extensions below |
| `docs/_diagrams/make_diagrams.py` | the diagrams |

A new public class or function must be added to the reference page of its
module (a module documented with `automodule` picks it up automatically).
Deprecated aliases are left out of autodoc and listed on
{doc}`../reference/deprecated`.

## Directives rendered from the code

| directive | extension | renders |
|---|---|---|
| `dualpath-table <module>` | `_ext/dualpath.py` | one row per operation of a module, pairing the `snake_case` autograd function with its `CamelCase` manual-backward twin |
| `autograd-only-table` | `_ext/dualpath.py` | `functions.dispatch.AUTOGRAD_ONLY`: the operations without a manual backward, and why |
| `deprecated-table` | `_ext/dualpath.py` | every object carrying the `__deprecated__` marker, with its replacement |
| `diagram <name>` | `_ext/diagrams.py` | `_static/diagrams/<name>.svg` in the light theme and `<name>_dark.svg` in the dark one; the content is the (parsed) caption |

The pairing logic is in `_ext/dualpath_pairing.py`, free of Sphinx imports so
that `tests/test_regressions.py` can test it.

## Diagrams

```bash
python docs/_diagrams/make_diagrams.py                 # all, light and dark
python docs/_diagrams/make_diagrams.py rresnet_manifold --png /tmp/previews
```

The script needs numpy and matplotlib (in the `docs` extra) and writes
deterministic SVG files (no timestamp, fixed element ids, fixed seeds):
regenerating an unchanged diagram is a no-op, so the files are committed. The
2 × 2 SPD matrices are computed exactly with the formulas of the package and
drawn as ellipses, tangent vectors as signed crosses; curved surfaces are
schematic. `assert` statements check the claims the figures make (unit
affine-invariant step, converged Karcher and Bures–Wasserstein means, centred
GBWBN batch of barycenter $I$, folding example outside the injectivity
domain).

| diagram | shows | used in |
|---|---|---|
| `spdnet_pipeline` | the layers of an SPDNet and the shape after each | landing page |
| `package_levels` | `model` / `nn` / `functions` / `random` and the neighbouring repositories | landing page, {doc}`../user_guide/concepts` |
| `reeig` | ReEig and ReEigBias as eigenvalue maps | {doc}`../user_guide/layers` |
| `parametrization` | static chart vs dynamic re-centring | {doc}`../user_guide/parametrizations` |
| `batchnorm_steps` | affine-invariant batch norm: Karcher mean, centring, scaling, bias | {doc}`../user_guide/batchnorm` |
| `gbwbn_steps` | the eight steps of GBWBN | {doc}`../user_guide/batchnorm` |
| `gbwbn_folding` | folding of $\mathrm{Exp}_I$: scalar picture, a folded matrix, scale and spread | {doc}`../user_guide/batchnorm` |
| `rresnet_pipeline` | the residual block, from $X_i$ to $X_{i+1}$ | {doc}`../user_guide/residual` |
| `rresnet_spectrum` | floored spectrum, signed field, learned frame $Q$ | {doc}`../user_guide/residual` |
| `rresnet_manifold` | tangent plane, AI unit ball, geodesic, whitened inset | {doc}`../user_guide/residual` |
| `rresnet_le` | the log-Euclidean block in log coordinates | {doc}`../user_guide/residual` |
| `rresnet_variants` | `branch_pt` and `congruence` | {doc}`../user_guide/residual` |
| `daleckii_krein` | autograd denominators vs divided differences | {doc}`../numerics/manual_backward` |
| `implicit_diff` | unrolled vs implicit differentiation | {doc}`../numerics/manual_backward` |

## Checks before a pull request

- `sphinx-build -W` passes (the CI job does the same on every pull request);
- the Python snippets of the getting-started and user-guide pages still run
  (they are written to run as-is, in order, page by page);
- new diagrams are checked visually in both themes.

# Development

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`git-pull-request` Contributing
:link: contributing
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Setup, branches, commits, code style, docstrings, pull requests and what the
CI checks.
:::

:::{grid-item-card} {octicon}`checklist` Testing
:link: testing
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

The test suite: gradchecks, manual vs autograd, the autograd-only graph walk,
float32 against float64, regressions and reference fixtures.
:::

:::{grid-item-card} {octicon}`book` Documentation
:link: documentation
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Building these pages, the directives that render tables from the code, the
diagrams and their light / dark variants.
:::

:::{grid-item-card} {octicon}`graph` Numerical benchmarks
:link: benchmarks
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

How the float32 / mixed-precision table and the timings were measured, and
how to rerun them.
:::
::::

```{toctree}
:hidden:

contributing
testing
documentation
benchmarks
```

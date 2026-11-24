---
html_theme.sidebar_secondary.remove: true
---

# yetanotherspdnet

```{raw} html
<div class="hero">
  <div>
    <p class="hero-title">yetanotherspdnet</p>
    <p class="hero-sub">Deep learning on SPD matrices, in PyTorch</p>
    <p class="tagline">SPDNet layers, Riemannian batch normalizations in seven
    geometries and Riemannian residual networks, built on SPD matrix functions
    whose hand-written gradients stay exact on ill-conditioned data.</p>
  </div>
  <div class="hero-logo"><img src="_static/logo.svg" alt="yetanotherspdnet logo"></div>
</div>
```

{bdg-primary-line}`Python ≥ 3.11` {bdg-primary-line}`PyTorch ≥ 2.0` {bdg-primary-line}`MIT licence`
{bdg-primary-line}`manual backward by default`

````{container} hero-buttons
```{button-ref} getting_started/installation
:ref-type: doc
:color: primary
:class: sd-px-4

{octicon}`rocket` Get started
```
```{button-ref} user_guide/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4

{octicon}`book` Read the guide
```
```{button-ref} reference/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4

{octicon}`code` API reference
```
````

```python
import torch
from yetanotherspdnet import SPDnet
from yetanotherspdnet.random.spd import random_SPD

X = random_SPD(n_features=93, n_matrices=32, cond=1e3, dtype=torch.float64)
model = SPDnet(
    input_dim=93, hidden_layers_size=[84, 63], output_dim=117, batchnorm=True
)
logits = model(X)  # (32, 117): BiMap -> ReEig -> BatchNorm -> ... -> LogEig -> Linear
```

```{diagram} spdnet_pipeline
:alt: The layers of an SPDNet on HDM05 and the shape after each of them

An SPDNet on HDM05 skeleton covariances: the matrices stay SPD through BiMap,
ReEig and the batch normalization; only LogEig maps them to a flat space for a
Euclidean classifier.
```

## Explore the documentation

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-start

Install the package (CPU or GPU PyTorch, extras) and train a first model in
ten lines.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: user_guide/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-guide

One page per topic: the mathematics, the implementation and the options of
the layers, geometries, batch normalizations, residual networks and models.
:::

:::{grid-item-card} {octicon}`pulse` Numerics
:link: numerics/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-numerics

Manual backward as the norm, divided differences, matrix equations, float32
and mixed precision, ill-conditioned data.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: reference/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-api

Every model, layer, parametrization and function, with the autograd and
manual-backward twins side by side, and the deprecated names.
:::

:::{grid-item-card} {octicon}`tools` Development
:link: development/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-dev

Contributing, the test suite (gradchecks, float32, regressions), the
documentation and its diagrams, the numerical benchmarks.
:::

:::{grid-item-card} {octicon}`mortar-board` References
:link: references
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-refs

The papers behind the layers, geometries and numerical methods, with their
DOI or arXiv identifier.
:::
::::

## Why yetanotherspdnet

::::{grid} 1 2 3 3
:gutter: 3
:class-container: features

:::{grid-item-card} {octicon}`shield-check` Exact gradients
Every eigendecomposition is differentiated by a hand-written backward
(Daleckii–Krein), finite at repeated eigenvalues, where autograd through
`eigh` is not. See {doc}`numerics/manual_backward`.
:::

:::{grid-item-card} {octicon}`globe` Seven geometries
Affine-invariant, log-Euclidean, arithmetic, harmonic, GAH and its adaptive
variant, Bures–Wasserstein: mean, geodesic, dispersion and transport behind
one registry. See {doc}`user_guide/geometries`.
:::

:::{grid-item-card} {octicon}`stack` Batch normalizations
Centring, scalar dispersion and an SPD bias in any geometry, and GBWBN with
its power deformation. See {doc}`user_guide/batchnorm`.
:::

:::{grid-item-card} {octicon}`git-branch` Residual networks
RResNet blocks with two eigendecompositions per block, and the options of
Res2SPDNet (transport branch, congruence). See {doc}`user_guide/residual`.
:::

:::{grid-item-card} {octicon}`cpu` float32 when it is enough
Mixed precision (`eigh_dtype=torch.float64`) and a measured table of where
float32 is safe. See {doc}`numerics/precision`.
:::

:::{grid-item-card} {octicon}`beaker` Robust covariance pooling
Tyler and Student-t M-estimators differentiated at their fixed point, as
network layers. See {doc}`user_guide/layers`.
:::
::::

## Where it fits

```{diagram} package_levels
:alt: The levels of the package and the neighbouring repositories

`model` uses `nn`, which uses `functions`; `random` generates SPD and
orthonormal matrices. Data come from
[spdnet-datasets](https://github.com/Yet-Another-Research-Organisation/spdnet-datasets);
[spdnet-training](https://github.com/Yet-Another-Research-Organisation/spdnet-training)
provides a Lightning module, a Hydra command line and CNN backbones with
covariance pooling.
```

```{admonition} Authors and licence
:class: tip

Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet. Released
under the MIT licence. When you publish results obtained with a layer, cite
the paper that introduced it ({doc}`references`).
```

```{toctree}
:hidden:

getting_started/index
user_guide/index
numerics/index
reference/index
development/index
references
```

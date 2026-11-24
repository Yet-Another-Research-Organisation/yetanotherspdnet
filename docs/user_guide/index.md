# User guide

One page per topic. Every page has the same three parts:

**Mathematics**
: the objects, formulas and properties, with the papers they come from
  ({doc}`../references`);

**Implementation**
: how the package computes them: manual backward or autograd, cost in
  eigendecompositions, dtype and mixed precision;

**Options and strategy**
: which argument to set, and when: static or dynamic parametrization,
  `use_autograd`, `eigh_dtype`, the geometry, $\theta$, the momentum, the
  step of the residual blocks…

Start with {doc}`concepts`, the map of the package; the other pages can be
read in any order.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb` Concepts
:link: concepts
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

The SPD manifold, the geometries at a glance, the three levels of the package,
the shapes inside a network and the conventions.
:::

:::{grid-item-card} {octicon}`stack` Layers
:link: layers
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

BiMap, ReEig, ReEigBias, LogEig, Vec / Vech, and the covariance layers
(sample covariance, M-estimators).
:::

:::{grid-item-card} {octicon}`sync` Parametrizations
:link: parametrizations
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Orthonormal, SPD and scalar parameters trained with a Euclidean optimizer:
static charts or dynamic trivializations.
:::

:::{grid-item-card} {octicon}`globe` Geometries
:link: geometries
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Affine-invariant, log-Euclidean, Kullback–Leibler, GAH, Bures–Wasserstein:
distance, mean, geodesic, transport, dispersion, and the registry.
:::

:::{grid-item-card} {octicon}`sliders` Batch normalization
:link: batchnorm
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Centring, scalar dispersion, SPD bias, running statistics; GBWBN with its
metric $M$, its power $\theta$ and the folding of its exponential.
:::

:::{grid-item-card} {octicon}`git-branch` Residual networks
:link: residual
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

The RResNet block (spectral field, unit step, exponential), its options, the
Res2SPDNet variants, and how the package differs from the paper and its code.
:::

:::{grid-item-card} {octicon}`package` Models
:link: models
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

`SPDnet`, `RResNet`, `GBWBNRResNet`: architectures, the constructor arguments
by role, and configurations of published experiments.
:::
::::

```{toctree}
:hidden:

concepts
layers
parametrizations
geometries
batchnorm
residual
models
```

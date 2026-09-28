# User guide

These pages explain the concepts behind the library and the choices that the
API reference does not spell out. Start with {doc}`concepts`, the map of the
package; the other pages can be read in any order.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} How the library fits together
:link: concepts
:link-type: doc
The three levels (functions, layers, models), the shapes flowing through a
network, and the conventions: float64, two gradient paths, parametrizations.
:::

:::{grid-item-card} Layers and parametrizations
:link: layers
:link-type: doc
BiMap, ReEig, LogEig and their equations; static vs dynamic parametrization
of the orthonormal and SPD parameters.
:::

:::{grid-item-card} Models and their options
:link: models
:link-type: doc
Which model to use, the constructor arguments grouped by role, and the
configurations of published experiments.
:::

:::{grid-item-card} Geometries
:link: geometries
:link-type: doc
The Riemannian geometries on the SPD manifold, their means and dispersions,
and how to pick one.
:::

:::{grid-item-card} Batch normalization
:link: batchnorm
:link-type: doc
How the SPD batch normalization layers centre, rescale and re-bias a batch,
and their training/evaluation behaviour.
:::

:::{grid-item-card} Residual networks
:link: residual
:link-type: doc
RResNet residual blocks: spectral vector field, affine-invariant (unit
step) or log-Euclidean exponential map.
:::

:::{grid-item-card} Numerical and optimization techniques
:link: numerics
:link-type: doc
Daleckii–Krein backwards, Sylvester equations, means as iterations, implicit
differentiation, parametrizations, batch normalization and residual-step
mechanics, precision — with their equations.
:::
::::

```{toctree}
:hidden:

concepts
layers
models
geometries
batchnorm
residual
numerics
```

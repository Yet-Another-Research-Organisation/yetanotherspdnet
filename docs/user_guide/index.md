# User guide

These pages explain the concepts behind the library and the choices that the
API reference does not spell out.

::::{grid} 1 1 3 3
:gutter: 2

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

:::{grid-item-card} Gradients and precision
:link: numerics
:link-type: doc
The `use_autograd` switch, hand-written backwards, and the float64 policy.
:::
::::

```{toctree}
:hidden:

geometries
batchnorm
numerics
```

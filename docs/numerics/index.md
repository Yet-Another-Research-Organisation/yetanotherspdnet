# Numerics

Training on the SPD manifold means differentiating eigendecompositions,
solving matrix equations inside backward passes, iterating to fixed points and
surviving very ill-conditioned data. Three rules hold everywhere in the
package:

1. **Manual backward is the norm.** With `use_autograd=False`, the default of
   every layer, model and parametrization, no matrix decomposition is
   differentiated by autograd; the few exceptions are listed, and reaching one
   emits an `AutogradOnlyWarning`, never a silent fallback.
2. **Gradients stay finite at repeated eigenvalues.** Daleckii–Krein
   backwards with closed-form divided differences, Sylvester equations with
   positive denominators, implicit differentiation at fixed points.
3. **Precision is measured, not assumed.** float64 by default; float32 and
   mixed precision where the measured table says they are safe.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`shield-check` Manual backward
:link: manual_backward
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

The two gradient paths, the list of operations without a manual backward,
Daleckii–Krein, divided differences, matrix equations, means, implicit
differentiation.
:::

:::{grid-item-card} {octicon}`cpu` Precision
:link: precision
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

float32, float64 or mixed precision (`eigh_dtype`): the recommendation and
the measured table, layer by layer and condition number by condition number.
:::

:::{grid-item-card} {octicon}`alert` Stability
:link: stability
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

What breaks on ill-conditioned, repeated or degenerate inputs, and the
floors, clamps and reformulations that prevent it.
:::
::::

```{toctree}
:hidden:

manual_backward
precision
stability
```

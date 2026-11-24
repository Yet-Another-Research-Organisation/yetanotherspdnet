# Parametrizations

`BiMap` weights must stay orthonormal, batch normalization biases (and the
GBWBN matrices $M$, $G$) SPD, scales positive. The package keeps them there
with `torch.nn.utils.parametrize` parametrizations, so that **any Euclidean
optimizer** (SGD, Adam…) trains them. API:
{doc}`../reference/parametrizations`.

## Mathematics

A *parametrization* (or *trivialization*) of a manifold $\mathcal M$ is a map
$\varphi : \mathbb{R}^d \to \mathcal M$: the optimizer updates an
unconstrained tensor $\Xi$, and the layer uses $\varphi(\Xi) \in \mathcal M$
{ref}`[Lezcano-Casado 2019] <ref-lezcano2019>`. A gradient step on $\Xi$ is a
step on $\mathcal M$ seen through the chart $\varphi$; it is a good
Riemannian step only where $\varphi$ is close to an isometry, typically near
its base point.

**Static.** $\varphi$ is fixed for the whole training: simple, but the chart
distorts the geometry more and more as the parameter moves away from where
$\varphi$ is well conditioned.

**Dynamic.** $\varphi$ is centred at a reference point $P_{ref}$, and every
$K$ optimizer steps the reference point jumps to the current value and the
unconstrained tensor is reset,

$$
P_{ref} \leftarrow \varphi_{P_{ref}}(\Xi), \qquad \Xi \leftarrow 0 ,
$$

so each step is a small move around the current point, where the chart is
accurate: a Riemannian optimizer implemented with a Euclidean one (a *dynamic
trivialization*, {ref}`[Lezcano-Casado 2019] <ref-lezcano2019>`).

```{diagram} parametrization
:alt: Static chart vs dynamic chart on a curved manifold
:width: 85%

Static: one chart around the initial weight, increasingly distorted. Dynamic:
the chart follows the weight, re-centred every `n_steps_ref_update` steps.
```

### Orthonormal weights (Stiefel manifold)

| | static | dynamic |
|---|---|---|
| trained tensor | $\Xi \in \mathbb{R}^{n_{in}\times n_{out}}$ | $\Xi$, projected on $T_{W_{ref}}\mathrm{St}$ |
| weight | `torch.nn.utils.parametrizations.orthogonal` (matrix exponential, Cayley or Householder) | $W = \mathrm{qf}\big(W_{ref} + P_{W_{ref}}(\Xi)\big)$ (QR retraction) or its polar factor |
| reference point | none | $W_{ref}$, moved every `n_steps_ref_update` steps |

with the projection on the tangent space
$P_W(\Xi) = \Xi - W\,\mathrm{sym}(W^\top \Xi)$,
$\mathrm{sym}(A) = \tfrac12(A + A^\top)$, and the retractions of
{ref}`[Absil et al. 2008] <ref-absil2008>`.

### SPD parameters

`SPDParametrization` maps a symmetric $S$ to $f(S)$, $f$ applied to the
eigenvalues: the exponential (`"exp"`) or the scaled softplus
$f(x) = \log_2(1 + 2^x)$ (`"softplus"`, the default), which satisfies
$f(0) = 1$: a zero parameter is the identity matrix. Its dynamic version,
`SPDAdaptiveParametrization`, works around a reference point $R$,

$$
B = R^{1/2}\, f\big(R^{-1/2} S R^{-1/2}\big)\, R^{1/2},
$$

with $R \leftarrow B$ every `n_steps_ref_update` steps: the chart is the
affine-invariant one centred at $R$.

### Scalars

Positive scalars (the scale $s$ of the batch normalizations) use the scaled
softplus (`ScalarSoftPlusParametrization`); scalars in $(0, 1)$ (the position
$t$ of the adaptive GAH mean) the sigmoid (`ScalarSigmoidParametrization`).

## Implementation

- **Manual backwards.** The SPD maps go through the Daleckii–Krein backward
  of the eigenvalue function (finite at the identity, where all eigenvalues of
  $S = 0$ are equal), the QR and polar retractions through hand-written
  backwards. The **static Stiefel parametrization** is torch's
  `orthogonal`, differentiated by autograd: it is on the list of operations
  without a manual backward and emits an `AutogradOnlyWarning` once
  ({doc}`../numerics/manual_backward`). Its backward has no eigenvalue gap,
  so this is a matter of policy, not of stability.
- **Cost.** An SPD parametrization costs one eigendecomposition of an
  $n \times n$ matrix per forward, independent of the batch size; the dynamic
  version also uses $R^{\pm 1/2}$, stored as buffers and recomputed only when
  $R$ moves.
- **Buffers.** The reference point and the last value of an adaptive
  parametrization are separate buffers (`clone()`d): a buffer sharing the
  storage of the reference point would silently move it at every forward
  pass.
- **The hook.** The jump of the reference point is an optimizer
  post-step hook (`recenter_dynamic_parametrization`), registered by
  `register_optimizer_hook(optimizer)` on the layer or the model.

## Options and strategy

| where | argument | values |
|---|---|---|
| `BiMap` | `parametrized`, `parametrization_mode`, `parametrization_options`, `n_steps_ref_update` | `True`; `"static"` / `"dynamic"`; static: options of `orthogonal` (`{"orthogonal_map": "cayley"}`…), dynamic: `{"mapping": "QR"}` or `{"mapping": "polar"}`; 100 |
| batch normalizations | `parametrization`, `parametrization_mode`, `n_steps_ref_update` | `"softplus"` / `"exp"`; `"static"` / `"dynamic"` (applies to the bias, and to $M$ and $G$ of GBWBN); 100 |
| residual blocks | `stiefel_parametrization_mode`, `stiefel_n_steps_ref_update` | the frame $Q$ of the spectral field |
| models | the same with a `bimap_`, `batchnorm_` or `stiefel_` prefix | |

- **Static** is the default and is enough for short trainings and small
  learning rates.
- **Dynamic** is the better choice for long trainings or large steps, and
  for SPD parameters that move far from the identity (GBWBN's $M$ and $G$
  on data whose scale is far from 1). `n_steps_ref_update` trades the accuracy
  of the chart (small) against the cost of the recentring (an
  eigendecomposition or a QR per move); 1 to 100 are reasonable.
- `"softplus"` grows linearly for large arguments, `"exp"` exponentially:
  prefer `"softplus"` unless the parameter must reach very small eigenvalues.

```{warning}
Without `register_optimizer_hook`, the reference point never moves and the
dynamic mode silently behaves like a static parametrization around the
initial value. The models forward the call to every dynamic layer:
`model.register_optimizer_hook(optimizer)`, once, after building the
optimizer.
```

```python
import torch

from yetanotherspdnet.nn import BiMap

layer = BiMap(n_in=64, n_out=32, parametrization_mode="dynamic", n_steps_ref_update=100)
optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
layer.register_optimizer_hook(optimizer)  # moves W_ref every 100 steps
```

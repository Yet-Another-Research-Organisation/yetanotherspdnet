# Residual networks

`RResNet` and `GBWBNRResNet` interleave BiMap layers with **residual blocks**
(`nn.rresnet_layers.ResidualBlock`) following
{ref}`[Katsman et al. 2023] <ref-katsman2023>`
([reference code](https://github.com/CUAI/Riemannian-Residual-Neural-Networks)).
The defaults reproduce the reference code; the other options, some taken from
Res2SPDNet {ref}`[Yue et al. 2026] <ref-yue2026>`, are listed below. API:
{doc}`../reference/residual`.

## Mathematics

### The residual step

A residual block moves each SPD matrix $X_i$ along a learned tangent vector,
the Riemannian analogue of $x + f(x)$:

$$
V(X_i) = Q \operatorname{diag}\big(f_\theta(\tilde\lambda_1, \dots, \tilde\lambda_n)\big) Q^\top,
$$

$$
X_{i+1} = \mathrm{Exp}_{X_i}\Big(\frac{V}{\lVert V \rVert_{X_i}}\Big)
= X_i^{1/2} \exp\big(\hat W\big) X_i^{1/2},
$$

where $\tilde\lambda_1 \le \dots \le \tilde\lambda_n$ are the (floored)
eigenvalues of $X_i$ in ascending order, $f_\theta : \mathbb{R}^n \to
\mathbb{R}^n$ a small Conv1d network along the sorted spectrum (with
BatchNorm1d; or an MLP), $Q$ a learned orthonormal matrix
(`SpectralVectorField`), $\lVert V \rVert_X = \lVert X^{-1/2} V X^{-1/2}
\rVert_F$ the affine-invariant norm and $\hat W = X_i^{-1/2} V X_i^{-1/2} /
\lVert V \rVert_{X_i}$. Every block moves by exactly one unit of
affine-invariant distance; $X_i$ is the base point of the step (the skip
connection), and no parallel transport is needed because $V$ is built and
applied at $X_i$.

```{diagram} rresnet_pipeline
:alt: The residual block as a pipeline, from X_i to X_{i+1}

The default affine-invariant block. One eigendecomposition (1) gives the
floored spectrum and $X_i^{\pm 1/2}$; the field $f_\theta$ and the learned
frame $Q$ give $V$; the unit step divides by its affine-invariant norm; the
exponential is the second eigendecomposition (2).
```

```{diagram} rresnet_spectrum
:alt: Eigenvalue view of the block: floored spectrum, signed field, learned frame

(a) The ascending spectrum of $X_i$, floored at $\epsilon$. (b) The field: a
Conv1d along the positions of the sorted spectrum (example weights), with
signed outputs. (c) On a 2 × 2 example, the learned frame $Q$ (arrows of
length $|f_k|$) is not the eigenbasis $U$ of $X_i$ (dotted).
```

```{diagram} rresnet_manifold
:alt: The affine-invariant step on the manifold, with the tangent plane and the unit ball

The step on the manifold (schematic surface, exact 2 × 2 matrices). The
tangent plane $T_{X_i}$ holds a slice of the affine-invariant unit ball
$\{\lVert X_i^{-1/2} V X_i^{-1/2} \rVert_F = 1\}$, an ellipse (green), against
the Frobenius unit circle (dashed); $\hat V$ ends on it. The geodesic
$t \mapsto X_i^{1/2} e^{t \hat W} X_i^{1/2}$ reaches $X_{i+1}$ at distance 1;
its points at $t = 0, \frac14, \frac12, \frac34, 1$ are drawn as ellipses.
Inset: after whitening ($X_i \mapsto I$) the ball is the unit circle and the
geodesic is $e^{t\hat W}$.
```

### Log-Euclidean block

With `residual_metric="log_euclidean"` (`metric` of the block), the step is
$X_{i+1} = \exp(\log X_i + V)$, $V$ unnormalized by default: a translation in
log coordinates. It is the log-Euclidean exponential
$\mathrm{Exp}^{LE}_{X_i}(V') = \exp(\log X_i + D\log_{X_i}[V'])$ of
$V' = D\exp_{\log X_i}[V]$, the field read as a displacement of $\log X$. The
reference code sends $\log X + V$ of the last block directly to the
classifier, without the exponential and the LogEig layer
(`le_output="log"`).

```{diagram} rresnet_le
:alt: The log-Euclidean block as a translation in log coordinates
:width: 75%

In the flat log coordinates (the plane spanned by $\log X_i$ and $V$, exact),
the block is a translation; the ellipses are $\exp(\log X_i + tV)$ for
$t = 0, \frac12, 1$.
```

### Why the step is normalized

$\lVert V \rVert_X$ grows like $1/\lambda_{\min}(X)$. On real covariances, for
instance HyperLeaf with eigenvalues from $10^{-4}$ to $3 \cdot 10^{3}$, an
unnormalized $V$ of Frobenius norm about 3 has an affine-invariant norm about
$2 \cdot 10^{3}$: the matrix exponential overflows, and the next
eigendecomposition fails. The unit step, like the log-Euclidean one, stays
finite for any conditioning (`TestResidualBlockIllConditioned`). It also bounds
the output: the eigenvalues of $e^{\hat W}$ lie in $[e^{-1}, e]$, so for any
$y$, $y^\top X_{i+1} y \in [e^{-1}, e]\, y^\top X_i y$, hence

$$
\lambda_{\min}(X_{i+1}) \ge e^{-1} \lambda_{\min}(X_i), \qquad
\lambda_{\max}(X_{i+1}) \le e\, \lambda_{\max}(X_i):
$$

the condition number grows by at most $e^2$ per block.

### Paper and reference code

The block follows the reference code, which differs from the paper on four
points:

- **Unit step.** The paper writes $\mathrm{Exp}_X(V(X))$; the code (design 4,
  HDM05) divides $V$ by its affine-invariant norm.
- **Ascending spectrum.** The Conv1d slides along the eigenvalues sorted in
  ascending order (`torch.linalg.eigh`): the field is a function of the
  position in the sorted spectrum, not a spectral function. At repeated
  eigenvalues it is not differentiable (and with `basis="eigenbasis"` not even
  defined), which does not occur on data.
- **Log-Euclidean translation.** $\exp(\log X + V)$ is a translation in log
  coordinates, not $\mathrm{Exp}^{LE}_X(V)$ (above).
- **Output clamp.** The code clamps the output eigenvalues to $[10^{-8},
  10^{8}]$ (`projx`); the bound above makes it unnecessary for the unit step
  (`output_clamp=True` restores it).

### Res2SPDNet variants

**Branch with parallel transport** (`residual="branch_pt"`). A branch
$Z = F(X)$ (default `BiMap(n, n)` then `ReEig`, or any module given as
`branch`), the field computed at $Z$, transported to $X$ by the
affine-invariant parallel transport and normalized at $X$ (the transport is an
isometry, so this equals the norm at $Z$):

$$
X_+ = \mathrm{Exp}_X\Big(\frac{\Gamma_{Z \to X} V(Z)}{\lVert V(Z) \rVert_Z}\Big),
$$

$$
\Gamma_{Z \to X}(V) = E V E^\top,\ E = (X Z^{-1})^{1/2} = X^{1/2} N^{-1/2} X^{-1/2},
$$

with $N = X^{-1/2} Z X^{-1/2}$. The Res2SPDNet code without transport
normalizes the field at $Z$ but applies it at $X$; that variant is not
offered. The log-Euclidean variant is $\exp(\log X + V(Z))$.

**Congruence** (`residual="congruence"`, the cheap SPD "addition" of
Res2SPDNet): $X_+ = X^{1/2} F(X) X^{1/2} = \mathrm{Exp}_X(\Gamma_{I \to X}
\log F(X))$ (`congruence_SPD_sqrtm(F(X), X)`). No vector field, no step.

The multi-granularity (hierarchical) architecture of Res2SPDNet is not
implemented.

```{diagram} rresnet_variants
:alt: The branch-with-transport and congruence variants

Left: `branch_pt`, the field computed at the branch output $Z$, transported to
$X$ and normalized, then the exponential at $X$. Right: `congruence`. Ellipses
and crosses are exact 2 × 2 computations.
```

## Implementation

**One fused decomposition.** $X = U \Lambda U^\top$ is decomposed once, with
the eigenvalues floored inside the decomposition,
$\tilde\lambda_i = \max(\lambda_i, \epsilon)$,
$\epsilon = \max(10^{-8}, \varepsilon \max_k \lambda_k)$ (`spd_floor`; the
floored matrix is never formed, so its small eigenvalues are exact). It gives
the spectrum of the field, $X^{\pm 1/2} = U \tilde\Lambda^{\pm 1/2} U^\top$ and
$\log X$, with one Daleckii–Krein backward for all of them
(`FlooredEighFunctions`). The norm is
$\lVert U^\top W U \rVert_F = \lVert \tilde\Lambda^{-1/2} (U^\top V U)
\tilde\Lambda^{-1/2} \rVert_F$ (no Cholesky factorization), and the exponential
of $\hat W$ is the second decomposition. On well-conditioned inputs the floor is
exactly the identity, values and gradients.

| block | eigendecompositions (forward; the backwards reuse them) |
|---|---|
| affine-invariant (default) | 2 (5 and a Cholesky in the previous implementation) |
| log-Euclidean (default) | 2 |
| log-Euclidean, `le_output="log"` | 1 |
| `basis="eigenbasis"` (AI or LE) | 1 |
| `output_clamp=True` | +1 |
| `residual="branch_pt"` (AI) | 4 + branch |
| `residual="congruence"` | 1 + branch |

Forward + backward on CPU (3 threads, batch 64, $n = 64$, float64):
affine-invariant block 64 → 30 ms, log-Euclidean block 40 → 25 ms, RResNet
with two blocks 145 → 76 ms against the previous implementation, which followed
the reference code.

**Numerically singular inputs.** After a batch normalization $X$ can be
numerically singular (folded GBWBN outputs, or the limit of float64 on very
ill-conditioned batches): the floor handles it inside the decomposition.

**Precision.** The default blocks are safe in float32 up to
$\kappa = 10^4$ and degraded (not failing) beyond; `branch_pt` forms
$N = X^{-1/2} Z X^{-1/2}$, whose condition number reaches $\kappa(X)\kappa(Z)$,
and is unsafe in float32 from $\kappa = 10^4$, even with
`eigh_dtype=torch.float64`; `basis="eigenbasis"` has float32 gradient errors
of order 1 at any conditioning (discontinuous at repeated eigenvalues),
float64 only ({doc}`../numerics/precision`).

## Options and strategy

`ResidualBlock` takes the options below; the models take them as
`residual_options={...}` (passed to every block; `le_output` only to the last
one; `branch` as a factory `n_features -> nn.Module`, called once per block).
With $\nu$ the norm of the field ($\lVert V \rVert_X$ for the affine-invariant
metric, $\lVert V \rVert_F$ for the log-Euclidean one):

| option | values (default first) | step |
|---|---|---|
| `metric` (`residual_metric` in the models) | `"affine_invariant"`, `"log_euclidean"` | $\mathrm{Exp}_X$, $\exp(\log X + \cdot)$ |
| `step` | reference: `"unit"` (AI), `"none"` (LE) | $V/\nu$, $V$ |
| | `"clip"` (`step_radius` $r$) | $V \min(1, r/\nu)$ |
| | `"learned_scale"` (`step_scale_init` $\alpha_0$) | $\alpha V/\nu$, $\alpha$ learned, $\alpha_0 = 10^{-2}$: near the identity at initialization |
| `spectrum_input` | `"raw"`, `"log"` | $f(\lambda)$, $f(\log\lambda)$ (a rescaling of $X$ shifts the input) |
| `basis` | `"learned_Q"`, `"eigenbasis"` | $V = Q \operatorname{diag}(f) Q^\top$, $V = U \operatorname{diag}(f) U^\top$ |
| `field_batchnorm` | `True`, `False` | BatchNorm1d between the Conv1d layers of $f$ |
| `residual` | `"exp"`, `"branch_pt"`, `"congruence"` | see above |
| `le_output` | `"spd"`, `"log"` | $\exp(\log X + V)$, $\log X + V$ (last block; LogEig skipped) |
| `output_clamp` | `None` (auto), `True`, `False` | eigenvalue clamp to $[\epsilon, 10^8]$; auto: on only for the unbounded AI steps `"none"` and `"learned_scale"` |

The field itself: `spectrum_type` (`"conv1d"` or `"mlp"`),
`spectrum_hidden_dim` (3), `spectrum_n_layers` (2), `spectrum_kernel_size` (5),
and the parametrization of $Q$ (`stiefel_parametrization_mode`,
`stiefel_n_steps_ref_update`, {doc}`parametrizations`).

- **Comparability with the paper:** keep the defaults (unit AI step, raw
  spectrum, learned $Q$, BatchNorm1d in the field); `"log_euclidean"` with
  `le_output="log"` is the variant that performs best on 3 of the 4 datasets
  of the paper.
- **Smaller steps.** The unit step is large at initialization; `step="clip"`
  bounds the step length, `step="learned_scale"` starts near the identity and
  learns it.
- **Scale-robust field.** `spectrum_input="log"` makes the field see
  $\log\lambda$, so a global rescaling of the data only shifts its input.
- **Cheaper blocks.** `basis="eigenbasis"` (one decomposition, an eigenvalue
  map) is experimental: float64 only.

```python
from yetanotherspdnet import RResNet

model = RResNet(
    input_dim=204,
    hidden_layers_size=[100, 50],
    n_residual_blocks=[1, 1],  # one block after each BiMap
    output_dim=4,
    residual_metric="log_euclidean",
    # as the reference code: the last block returns log X + V, no LogEig
    residual_options={"le_output": "log"},
)
```

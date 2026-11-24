# Riemannian residual networks

`RResNet` and `GBWBNRResNet` interleave BiMap layers with **residual blocks**
(`nn.rresnet_layers.ResidualBlock`), following Katsman et al., *Riemannian
Residual Neural Networks*, NeurIPS 2023 ([arXiv:2310.10013](https://arxiv.org/abs/2310.10013),
[reference code](https://github.com/CUAI/Riemannian-Residual-Neural-Networks)).
The default options reproduce the reference code; the other options, some
taken from Res2SPDNet (Yue et al., CVPR 2026 Findings,
[code](https://github.com/tewubieiDD/Res2SPDNet)), are listed below.

## The residual step

A residual block moves each SPD matrix $X$ along a learned tangent vector,
the Riemannian analogue of $x + f(x)$:

$$
V(X) = Q \operatorname{diag}\big(f(\lambda_1, \dots, \lambda_n)\big) Q^\top,
$$

where $\lambda_1 \le \dots \le \lambda_n$ are the eigenvalues of $X$ in
ascending order, $f : \mathbb{R}^n \to \mathbb{R}^n$ a small Conv1d network
along the sorted spectrum (with BatchNorm1d; or an MLP) and $Q$ a learned
orthogonal matrix (`SpectralVectorField`). The step is chosen with
`residual_metric`:

`"affine_invariant"` (default)
: $X_+ = \mathrm{Exp}_X\big(V / \lVert V \rVert_X\big)
  = X^{1/2} \exp\big(W / \lVert W \rVert_F\big) X^{1/2}$, with
  $W = X^{-1/2} V X^{-1/2}$ and $\lVert V \rVert_X = \lVert W \rVert_F$:
  every block moves by exactly one unit of affine-invariant distance.

`"log_euclidean"`
: $X_+ = \exp\big(\log X + V\big)$, the variant that performs best on 3 of
  the 4 datasets of the paper.

### What is computed

Each block computes two eigendecompositions (five and a Cholesky
factorization before; see {doc}`numerics` for the timing):

1. $X = U \Lambda U^\top$, with the eigenvalues floored,
   $\tilde\lambda_i = \max(\lambda_i, \epsilon)$ (see below). This single
   decomposition gives the spectrum of the field, $X^{1/2} = U \tilde\Lambda^{1/2}
   U^\top$ and $X^{-1/2}$ (affine-invariant), or $\log X$ (log-Euclidean),
   with one hand-written Daleckii–Krein backward for all of them
   (`FlooredEighFunctions`).
2. The exponential: $W = X^{-1/2} V X^{-1/2}$ (in the eigenbasis of $X$,
   $U^\top W U = \tilde\Lambda^{-1/2} (U^\top V U) \tilde\Lambda^{-1/2}$),
   $\nu = \lVert W \rVert_F$ (no Cholesky factorization), and
   $X_+ = X^{1/2} \exp(W/\nu) X^{1/2}$ with one decomposition of $W$; or
   $\exp(\log X + V)$.

The reference code clamps the eigenvalues of the output to $[10^{-8},
10^{8}]$ (`projx`). The unit step does not need it: the eigenvalues of
$e^{W/\nu}$ lie in $[e^{-1}, e]$, so for any vector $y$,
$y^\top X_+ y = (X^{1/2} y)^\top e^{W/\nu} (X^{1/2} y) \in [e^{-1}, e]\,
y^\top X y$ and

$$
\lambda_{\min}(X_+) \ge e^{-1} \lambda_{\min}(X), \qquad
\lambda_{\max}(X_+) \le e\, \lambda_{\max}(X):
$$

the output is SPD with a condition number at most $e^2$ times that of the
input. The clamp is therefore off by default (`output_clamp=None`: on only
for the unbounded affine-invariant steps `"none"` and `"learned_scale"`); it
acted only when $\lambda_{\min}(X) < e \cdot 10^{-8}$ or $\lambda_{\max}(X)
> 10^{8}/e$, and the next block floors its input anyway.

### Paper and reference code

The block follows the reference code, which differs from the paper on four
points:

- **Unit step.** The paper writes $\mathrm{Exp}_X(V(X))$; the code (design 4,
  HDM05) divides $V$ by its affine-invariant norm. Without it,
  $\lVert V \rVert_X$ grows like $1/\lambda_{\min}(X)$ (see below).
- **Ascending spectrum.** The Conv1d slides along the eigenvalues sorted in
  ascending order (`torch.linalg.eigh`): the field is a function of the
  position in the sorted spectrum, not a spectral function. At repeated
  eigenvalues it is not differentiable (and with `basis="eigenbasis"` not
  even defined), which does not occur on data.
- **Log-Euclidean translation.** $\exp(\log X + V)$ is a translation in log
  coordinates, not the log-Euclidean exponential map
  $\mathrm{Exp}^{LE}_X(V) = \exp(\log X + D\log_X[V])$
  (`spd_geometries.log_euclidean.log_euclidean_exp`, with its inverse
  `log_euclidean_log`): it is $\mathrm{Exp}^{LE}_X(V')$ for
  $V' = D\exp_{\log X}[V]$, the field read as a displacement of $\log X$.
  The reference code also sends $\log X + V$ of the last block directly to
  the classifier, without the exponential and the LogEig layer: this is
  `le_output="log"`.
- **Output clamp.** `projx` (see above), here `output_clamp=True`.

## Options

`ResidualBlock` takes the options below; the models take them as
`residual_options={...}` (passed to every block; `le_output` only to the
last one). With $\nu$ the norm of the field ($\lVert V \rVert_X$ for the
affine-invariant metric, $\lVert V \rVert_F$ for the log-Euclidean one):

| option | values (default first) | step |
|---|---|---|
| `step` | reference: `"unit"` (AI), `"none"` (LE) | $V/\nu$, $V$ |
| | `"clip"` (`step_radius` $r$) | $V \min(1, r/\nu)$ |
| | `"learned_scale"` (`step_scale_init` $\alpha_0$) | $\alpha V/\nu$, $\alpha$ learned, $\alpha_0 = 10^{-2}$: near the identity at initialization |
| `spectrum_input` | `"raw"`, `"log"` | $f(\lambda)$, $f(\log\lambda)$ (a rescaling of $X$ shifts the input) |
| `basis` | `"learned_Q"`, `"eigenbasis"` | $V = Q \operatorname{diag}(f) Q^\top$, $V = U \operatorname{diag}(f) U^\top$ |
| `field_batchnorm` | `True`, `False` | BatchNorm1d between the Conv1d layers of $f$ |
| `residual` | `"exp"`, `"branch_pt"`, `"congruence"` | see below |
| `le_output` | `"spd"`, `"log"` | $\exp(\log X + V)$, $\log X + V$ (last block; LogEig skipped) |
| `output_clamp` | `None` (auto), `True`, `False` | eigenvalue clamp to $[\epsilon, 10^8]$ |

**Eigenbasis field** (`basis="eigenbasis"`). On the eigenvectors of $X$
itself, $V = U \operatorname{diag}(f) U^\top$ commutes with $X$ and the step
is an eigenvalue map, with one eigendecomposition per block:

$$
\text{AI:}\quad W = U \operatorname{diag}(f_i/\lambda_i) U^\top,\
\nu = \big\lVert (f_i / \lambda_i)_i \big\rVert_2,\
X_+ = U \operatorname{diag}\big(\lambda_i\, e^{s f_i/\lambda_i}\big) U^\top;
\qquad
\text{LE:}\quad X_+ = U \operatorname{diag}\big(\lambda_i\, e^{s f_i}\big) U^\top,
$$

with the step factor $s$ ($1/\nu$ for the unit step). Its backward splits the
dependence on the eigenvalues (through $f$) from the rotation of the
eigenvectors, $\Omega_{ij} = (g_i - g_j)/(\lambda_i - \lambda_j)$ for the new
eigenvalues $g$.

**Branch with parallel transport** (`residual="branch_pt"`, Res2SPDNet).
A branch $Z = F(X)$ (default `BiMap(n, n)` then `ReEig`, or any module given
as `branch`), the field computed at $Z$, transported to $X$ by the
affine-invariant parallel transport and normalized at $X$ (the transport is
an isometry, so this equals the norm at $Z$):

$$
X_+ = \mathrm{Exp}_X\Big(\frac{\Gamma_{Z \to X} V(Z)}{\lVert V(Z) \rVert_Z}\Big),
\qquad
\Gamma_{Z \to X}(V) = E V E^\top,\ E = (X Z^{-1})^{1/2} = X^{1/2} N^{-1/2} X^{-1/2},
$$

with $N = X^{-1/2} Z X^{-1/2}$: in whitened coordinates
$X^{-1/2}\Gamma(V)X^{-1/2} = N^{-1/2} (X^{-1/2} V X^{-1/2}) N^{-1/2}$. The
Res2SPDNet code without transport normalizes the field at $Z$ but applies it
at $X$; that variant is not offered. The log-Euclidean variant is
$\exp(\log X + V(Z))$ (the log-Euclidean transport is the identity in log
coordinates). `affine_invariant_parallel_transport` accepts one reference
pair per matrix. In float32, $N$ has a condition number up to
$\kappa(X)\kappa(Z)$: use float64 beyond $\kappa \approx 10^2$.

**Congruence** (`residual="congruence"`, the cheap SPD "addition" of
Res2SPDNet): $X_+ = X^{1/2} F(X) X^{1/2}$, with $X^{1/2}$ from the
decomposition of $X$; it equals
$\mathrm{Exp}_X(\Gamma_{I \to X} \log F(X))$ and `congruence_SPD_sqrtm(F(X), X)`.
No vector field, no step.

The multi-granularity (hierarchical) architecture of Res2SPDNet is not
implemented.

### Cost

Eigendecompositions of one forward pass (the backwards reuse them):

| block | before | now |
|---|---|---|
| affine-invariant (default) | 5 + Cholesky | 2 |
| log-Euclidean (default) | 4 | 2 |
| log-Euclidean, `le_output="log"` | – | 1 |
| `basis="eigenbasis"` (AI or LE) | – | 1 |
| `output_clamp=True` | (always on, included above) | +1 |
| `residual="branch_pt"` (AI) | – | 4 + branch |
| `residual="congruence"` | – | 1 + branch |

## Numerically singular inputs

After a batch normalization, $X$ can be numerically singular: the
Bures–Wasserstein steps of GBWBN fold matrices that leave the injectivity
domain of their exponential (see {doc}`batchnorm`), and very ill-conditioned
batches reach the limit of float64 precision. The block floors the
eigenvalues of its input at $\epsilon = \max(10^{-8}, \varepsilon
\max\lambda)$ (`spd_floor`; $10^{-8}$ in float64 at moderate scale), inside
the decomposition of step 1: the floored matrix is never formed, so its small
eigenvalues are exact instead of carrying the error $\varepsilon \lVert X
\rVert$ of a second decomposition. On well-conditioned inputs the floor is
exactly the identity, values and gradients.

## Why the normalization matters

$\lVert V \rVert_X$ grows like $1 / \lambda_{\min}(X)$. On real covariances,
for instance HyperLeaf with eigenvalues from $10^{-4}$ to $3 \cdot 10^{3}$,
an unnormalized $V$ of Frobenius norm about 3 has an affine-invariant norm
about $2 \cdot 10^{3}$. The matrix exponential then overflows, and the next
eigendecomposition fails. The normalized step, like the log-Euclidean one,
stays finite for any conditioning. `tests/nn/test_rresnet_layers.py`
(`TestResidualBlockIllConditioned`) checks this on matrices spanning that
range. `step="clip"` keeps the steps shorter than $r$ and
`step="learned_scale"` learns their length.

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

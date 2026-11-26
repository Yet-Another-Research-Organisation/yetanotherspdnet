# Batch normalization

Euclidean batch normalization subtracts the batch mean, divides by the batch
standard deviation, then applies a learned scale and shift. The layers of
`yetanotherspdnet.nn.batchnorm` do the same on the SPD manifold, in the
geometry chosen with `mean_type` ({doc}`geometries`). API:
{doc}`../reference/batchnorm`.

| layer | steps | origin |
|---|---|---|
| `BatchNormSPDMean` | centre the batch at its mean, move it to a learned SPD bias | {ref}`[Brooks et al. 2019] <ref-brooks2019>` |
| `BatchNormSPDMeanScalarVariance` | centre, rescale by a scalar dispersion, bias | {ref}`[Kobler et al. 2022] <ref-kobler2022>` |
| `BatchNormSPDMeanScalarVariance(mean_type="bures_wasserstein")` | GBWBN: power deformation, Bures–Wasserstein centring, scaling and bias | {ref}`[Wang et al. 2025] <ref-wang2025>` |

## Mathematics

### Mean only

With $\bar X$ the batch mean in the chosen geometry and $G$ the learned SPD
bias (`Covbias`), the default `norm_and_bias_mode="whitening"` is

$$
X_i \longmapsto G\,\bar X^{-1/2} X_i \bar X^{-1/2}\,G :
$$

whitening by $\bar X^{-1/2}$ (a congruence, so affine-invariant distances
within the batch are preserved and the Karcher mean becomes $I$), then a
congruence by $G$, which sends the mean to $G^2$.
`norm_and_bias_mode="parallel_transport"` transports the batch from $\bar X$ to
$G$ in the geometry of the mean, so the mean lands on $G$ itself:
affine-invariant $X \mapsto E X E^\top$,
$E = G^{1/2}(G^{-1/2}\bar X G^{-1/2})^{-1/2}G^{-1/2}$; log-Euclidean
$X \mapsto \exp(\log X - \log \bar X + \log G)$.

```{diagram} batchnorm_steps
:alt: Affine-invariant batch normalization of 2x2 matrices, step by step

2 × 2 SPD matrices drawn as ellipses, computed exactly: the batch and its
Karcher mean (iterated to a zero gradient), the centred batch (Karcher mean
$I$), rescaled along the geodesics from $I$ by $s/\sigma$, then biased by the
congruence with $G$ (mean $G^2$).
```

### Scalar dispersion

`BatchNormSPDMeanScalarVariance` inserts, between centring and bias, a
rescaling of the centred matrices along the geodesics from $I$:
$C \mapsto C^{s/\sigma}$ (affine-invariant), with $\sigma$ the batch dispersion
of the geometry (a scalar, $\sigma^2 = \frac1N \sum_i d(\bar X, X_i)^2$ or the
divergence of {doc}`geometries`) and $s > 0$ learned (`stdScalarbias`). The
centred batch then has dispersion $s$.

### GBWBN

With `mean_type="bures_wasserstein"`, the layer is the generalized
Bures–Wasserstein batch normalization {ref}`[Wang et al. 2025] <ref-wang2025>`
(their Algorithm 1). A learned SPD metric $M$ and a power $\theta \ne 0$
deform the data, the Bures–Wasserstein steps act in the deformed space, and
the result is mapped back:

1. deformation: $\hat X_i = M^{-1/2} X_i^{\theta} M^{-1/2}$;
2. batch barycenter $B$ of the $\hat X_i$ (Bures–Wasserstein fixed point);
3. centring: $C_i = \mathrm{Exp}_I\big(\Gamma_{B\to I}\,\mathrm{Log}_B(\hat X_i)\big)$;
4. scaling at $I$: $\mathrm{Exp}_I\big(\tfrac{s}{\sqrt{\sigma^2 + 10^{-5}}}\,\mathrm{Log}_I(C_i)\big)$,
   with $\sigma^2 = \frac{1}{\theta^2} \frac1N \sum_i d_{BW}(B, \hat X_i)^2$,
   the variance of the deformed metric;
5. bias: $\mathrm{Exp}_{\hat G}\big(\Gamma_{I\to\hat G}\,\mathrm{Log}_I(\cdot)\big)$,
   with $\hat G = M^{-1/2} G^{\theta} M^{-1/2}$ for the learned bias $G$
   (input space);
6. inverse deformation: $Y_i = \big(M^{1/2} (\cdot) M^{1/2}\big)^{1/\theta}$.

The Bures–Wasserstein formulas are those of {doc}`geometries`; the transports
are the exact Levi-Civita transports between $I$ and a point.

```{diagram} gbwbn_steps
:alt: The steps of GBWBN on 2x2 matrices

GBWBN on a batch of six 2 × 2 matrices, $\theta = 1/2$, computed with the
formulas of the package. Ellipses: SPD matrices (orange, the reference point of
each step); crosses: tangent vectors, one segment per eigenvector (blue
positive, red negative eigenvalue). The script checks that the centred batch
has barycenter exactly $I$ (no folding here).
```

### Folding: why the data scale matters

$\mathrm{Exp}_I(V) = (I + V/2)^2$ is injective only on $I + V/2 \succ 0$.
Outside, the matrix is *folded*: $V$ and the vector $V'$ with the same
eigenvectors and $|1 + v_k/2|$ in place of $1 + v_k/2$ give the same point, the
centred batch no longer has mean $I$, and a matrix close to the boundary
($1 + v_k/2 \approx 0$) comes out nearly singular. The Bures–Wasserstein metric
is not scale invariant: for data of scale $c$, the centring transports a
tangent vector of size $\sim c$ from a mean of size $\sim c$ to $I$, which
gives a vector of size $\sim \sqrt c$. **Folding is driven by how far the
eigenvalues are from 1** (their common scale and their spread), not by the
condition number as such.

```{diagram} gbwbn_folding
:alt: Folding of the Bures-Wasserstein exponential at the identity

(a) $\mathrm{Exp}_I$ on one eigenvalue: beyond $v = -2$, two tangent vectors
give the same point. (b) Data multiplied by 100, $\theta = 1$: the centring of
one matrix leaves the injectivity domain. (c) Smallest eigenvalue of
$I + V_i/2$ at the centring step against the scale $c$ of the data: without
initialization, folding starts at $c \approx 5$ ($\theta = 1$) or
$c \approx 500$ ($\theta = 1/2$) for this batch; `bw_init_metric="trace_scale"` removes the
dependence on $c$, and only a large spread ($\kappa \approx 10^4$) still folds,
at $\theta = 1$ but not at $\theta = 1/2$.
```

### The bias step: projection onto the domain of $\mathrm{Exp}_G$

The bias step $\mathrm{Exp}_{\hat G}(V) = T \hat G T$, with $T = I + Z$ and
$\hat G Z + Z \hat G = V$, has the same domain: $T$ is the optimal (Brenier)
transport map from $\hat G$ to the output, positive definite inside the domain.
The transport $\Gamma_{I \to \hat G}$ is an isometry but does not preserve
the domain: in the eigenbasis of $\hat G$ (eigenvalues $\delta_i$),
$Z_{ij} = \mathrm{Log}_I(X)_{ij} / \sqrt{2(\delta_i + \delta_j)}$, so a bias
with small eigenvalues and a large scaling $s / \sigma$ push $T$ to the
boundary, where the output is singular.
In training this happens without any fold of the centring: the second GBWBN of
a 2-BiMap SPDnet on HDM05 reached $\lambda_{\min}(T) = 8 \cdot 10^{-5}$
($\lambda_{\min}(\hat G) = 0.07$, $s/\sigma = 4.9$), an output of
$\kappa \approx 10^{16}$ after the power $1/\theta$, and a negative
eigenvalue in `LogEig` (NaN loss). The layer therefore projects $T$ onto
$\{T \succeq \tau I\}$ (eigenvalues clamped at $\tau$, manual backward)
with $\tau = \varepsilon^{|\theta|/4}$ ($\varepsilon$ the float64 machine
epsilon: $1.2 \cdot 10^{-2}$ at $\theta = 1/2$, $1.2 \cdot 10^{-4}$ at
$\theta = 1$, see `bw_transport_floor`): a direction at the bound
shrinks the output by at most $\tau^{2/|\theta|} = \sqrt{\varepsilon}$
relative to $\hat G$. The projection is exactly the identity (bit for bit) when
$T \succ \tau I$. The centring and the scaling steps are not projected.

**Precision.** GBWBN computes in float64 whatever the dtype of its data
(`BW_COMPUTE_DTYPE`): the data, $M$, $G$, $s$ and the running statistics are
cast to float64 on the fly, every decomposition is computed in float64, and the
output is cast back to the input dtype. Parameters and buffers keep the dtype
of the layer (state dicts are unchanged) and float32 parameters receive float32
gradients through the casts. In a float32 model the layer therefore matches the
float64 layer up to the rounding of its input and output, at the cost of
float64 arithmetic for this layer only; the bound $\tau$ above is always the
float64 one (float32 would need $\tau \approx 0.14$ at $\theta = 1/2$).

## Implementation

- **Manual backwards everywhere** (`use_autograd=False`, the default): the
  means, the whitening (a Sylvester equation in the backward), the transports,
  the Lyapunov solve of $\mathrm{Exp}_B$ and the polar factors of the
  Bures–Wasserstein cross terms are hand-written backwards. They stay finite
  where autograd through `eigh` is not: at the identity, where the bias, $M$
  and $G$ start, and on batches of scalar matrices $cI$
  ({doc}`../numerics/manual_backward`).
- **Cost.** One mean (see {doc}`geometries`), one $\bar X^{-1/2}$ (one
  eigendecomposition of an $n \times n$ matrix) and one congruence per matrix
  for the mean-only layer; the scalar variance adds one eigendecomposition per
  matrix (the power $C^{s/\sigma}$). GBWBN: $X_i^\theta$, the square roots of
  the barycenter iteration (one SVD per matrix and iteration), the transports
  (two eigendecompositions of $n \times n$ matrices), $\mathrm{Exp}_I$ and
  $\mathrm{Log}_I$ (one square root per matrix) and the inverse power.
- **Running statistics** follow the geometry: $\bar X_r \leftarrow \bar X_r
  \#_\mu \bar X$ on the geodesic of the geometry (step `momentum` $\mu$) and
  $\sigma_r^2 \leftarrow (1-\mu)\sigma_r^2 + \mu\,\sigma^2$. GBWBN stores its
  running mean **in the input space**, $B_r = (M^{1/2} \hat B_r
  M^{1/2})^{1/\theta}$, and maps it with the current $M$ when it is used, so it
  is not left in the coordinates of an old $M$ when $M$ is trained. This is
  exact when $M$ changes by a factor (the Bures–Wasserstein mean is
  1-homogeneous); for a general change of $M$ it is the natural convention,
  not an invariance.
- **Checkpoints.** GBWBN state dicts saved before that change (version 1) are
  converted when loaded (running mean mapped to the input space with the
  checkpoint's own $M$, missing reference points of dynamic layers set to $I$),
  so old checkpoints load with `strict=True`. New state dicts carry a
  `bw_state_version` buffer and are never converted twice.
- **Guards.** The exponent $s/\sigma$ uses $\max(\sigma, 3.2 \cdot 10^{-3})$,
  as bounded as the GBWBN factor $s/\sqrt{\sigma^2 + 10^{-5}}$, and the
  standard deviations are $\sqrt{\max(v, 0)}$: a batch of identical matrices
  gives finite outputs and gradients.

| mode | statistics used | running statistics |
|---|---|---|
| `model.train()`, batch of ≥ 2 matrices | batch mean (and dispersion) | updated with `momentum` |
| `model.train()`, a single matrix | running mean (and dispersion) | **not** updated, a `UserWarning` is emitted |
| `model.eval()` | running mean (and dispersion) | not updated |

A batch of one matrix has no statistics (its mean is itself, its dispersion
zero); normalizing with them would map every input to the identity. Use
`drop_last=True` in the `DataLoader` to avoid such batches.

### Package, paper and reference code (GBWBN)

| | paper (Algorithm 1) | reference code (jjscc/GBWBN) | this package |
|---|---|---|---|
| variance | to the batch mean | to the identity (`cal_var`) | to the batch mean |
| $M$, $G$ | Riemannian SGD | Riemannian SGD, mixing affine-invariant and BW updates | softplus / exp parametrization, any PyTorch optimizer, static or dynamic chart |
| gradient through the batch statistics | not specified | none | yes (`bw_batch_stats_grad=True`), optional |
| $\theta$ | hyper-parameter (ablation in Table 10) | class attribute shared by all layers (`Power_op._power`), which corrupts its gradient | per layer, validated |
| running-mean momentum | — | 0.1 | 0.01 for the layer, 0.1 in `GBWBNRResNet` |
| running mean | — | transformed space | input space |
| cross term $(B^{1/2}XB^{1/2})^{1/2}$ | — | eigendecomposition | SVD (polar factor) |

The conformity test of the package (`TestGBWBNConformity`) compares the layer
with the paper's formulas, not with the reference code.

## Options and strategy

`mean_type`
: The geometry ({doc}`geometries`). `"affine_invariant"` for fidelity,
  `"geometric_arithmetic_harmonic"` (default of `SPDnet`) or
  `"log_euclidean"` for speed, `"bures_wasserstein"` with the scalar variance
  for GBWBN. `mean_options={"n_iterations": 5}` for the iterative means.

`momentum`
: Step of the running statistics on the geodesic (0.01 by default). Larger
  values (0.1) follow a drifting distribution faster but are noisier.

`norm_strategy`, `minibatch_mode`, `minibatch_momentum`, `minibatch_maxstep`
: `"classical"` normalizes each batch with its own mean. `"minibatch"`
  normalizes with a smoothed mean
  $m_k = m_{k-1} \#_{\mu_k} \bar X_k$ ($\#$ the geodesic of the geometry),
  which reduces the noise of small batches; $\mu_k$ is `"constant"`, decays
  towards (`"decay"`) or grows from (`"growth"`) `minibatch_momentum` over
  `minibatch_maxstep` steps.

`norm_and_bias_mode`
: `"whitening"` (default, one $\bar X^{-1/2}$, mean sent to $G^2$) or
  `"parallel_transport"` (mean sent to $G$, distances preserved in the
  geometry; not for `"bures_wasserstein"` nor for the scalar-variance layer).
  The two parametrize the bias differently ($G^2$ versus $G$), so a trained
  `Covbias` is not interchangeable. With `use_autograd=True` the
  affine-invariant and log-Euclidean transports raise a `ValueError`: they take
  square roots or logarithms of the bias, which starts at the identity, where
  autograd through `eigh` gives NaN gradients.

`parametrization`, `parametrization_mode`, `n_steps_ref_update`
: How the bias (and $M$, $G$) stay SPD ({doc}`parametrizations`). The
  `"dynamic"` mode needs `layer.register_optimizer_hook(optimizer)`.

GBWBN: `bw_theta`, `bw_batch_stats_grad`, `bw_init_metric`
: In the models, `batchnorm_bw_options={"bw_theta": 0.25, ...}`.

  - `bw_theta` (default 0.5): power deformation $X \mapsto X^\theta$. It must
    be finite and nonzero ($X^0 = I$ erases the data). $\theta < 1$
    compresses the spread of the eigenvalues ($[10^{-2}, 10^{2}] \to
    [10^{-1}, 10]$ at $\theta = 1/2$), which reduces folding. The best value
    depends on the dataset: 0.5 on HDM05 and 0.75 on NTU in the paper's
    ablation (Table 10); 0.5 is also the reference code's default.
  - `bw_batch_stats_grad` (default `True`): gradients flow through the batch
    mean and variance. `True` is a choice of this package, not a prescription
    of the paper; `False` treats them as constants, as the reference code.
  - `bw_init_metric` (default `"identity"`): `"trace_scale"` sets, on the
    first training batch, $M = cI$ and $G = c^{1/\theta} I$ with
    $c = \operatorname{tr}(\bar B)/n$, $\bar B$ the barycenter of the
    $X_i^\theta$: the deformed data $X_i^\theta / c$ have unit scale and
    $\hat G = I$. The layer is then scale-equivariant at initialization
    ($f(sX) = s f(X)$), and its outputs keep the scale of the data.

**Against folding.** Measured on random batches (20 batches of 32), fraction
of folded matrices:

| data ($\kappa$, eigenvalues) | $n$ | $\theta$ | `"identity"` | `"trace_scale"` |
|---|---|---|---|---|
| $10^2$, $[10^{-2}, 1]$ (and $\times 10^{-2}$) | 8 | 1 and 0.5 | 0 | 0 |
| $10^2$, $[1, 10^{2}]$ | 8 | 1 | 99.8 % | 0 |
| $10^2$, $[1, 10^{2}]$ | 8 | 0.5 | 1.1 % | 0 |
| $10^2$, $[10^{2}, 10^{4}]$ | 8 | 0.5 | 98.3 % | 0 |
| $10^2$, $[10^{-1}, 10]$ | 64 | 1 | 22 % | 0 |
| $10^4$, $[10^{-2}, 10^{2}]$ | 64 | 0.5 | 22 % | 0 |
| $10^4$, $[10^{-2}, 10^{2}]$ | 64 | 1 | 100 % | 100 % |
| $10^6$, $[10^{-3}, 10^{3}]$ | 64 | 0.5 | 100 % | 57 % |

Initializing $M$ alone would move the folding to the bias step (97 % at scale
$10^2$, $\theta = 1$). In practice: rescale the data or use
`bw_init_metric="trace_scale"`, keep $\theta < 1$ when the spectrum is wide,
and keep `use_autograd=False` ($M$ and $G$ start at the identity, where the
manual backwards are finite). The layer computes in float64 even in a float32
model, so folded matrices close to singular keep their digits
({doc}`../numerics/precision`).

```python
import torch

from yetanotherspdnet.nn import BatchNormSPDMeanScalarVariance
from yetanotherspdnet.random.spd import random_SPD

gbwbn = BatchNormSPDMeanScalarVariance(
    n_features=8,
    mean_type="bures_wasserstein",
    bw_theta=0.5,
    bw_init_metric="trace_scale",
    parametrization_mode="dynamic",
)
optimizer = torch.optim.Adam(gbwbn.parameters(), lr=1e-3)
gbwbn.register_optimizer_hook(optimizer)
X = 100 * random_SPD(n_features=8, n_matrices=32, cond=100, dtype=torch.float64)
print(gbwbn(X).shape)  # torch.Size([32, 8, 8])
```

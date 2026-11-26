# Precision

float64 is the default of every layer and model. float32 halves the memory and
is faster, but eigendecompositions and divided differences lose digits quickly
on ill-conditioned matrices. This page gives the measured behaviour of every
layer in float32 and in mixed precision, and a recommendation.

## Recommendation

| your data (condition number $\kappa$ of the inputs) | use |
|---|---|
| $\kappa \le 10^{2}$ | float32 for every layer and model |
| $\kappa \le 10^{4}$ | float32 for `SPDnet`, `RResNet`, `GBWBNRResNet`, the residual blocks and the batch normalizations (the GBWBN layer computes in float64 internally), except `residual="branch_pt"` (float64); mixed precision (`eigh_dtype=torch.float64`) for a margin |
| $\kappa \le 10^{6}$ | mixed precision for `SPDnet`, `RResNet` and ReEig (safe); float64 for the batch normalizations (degraded or unsafe in float32) and for `GBWBNRResNet` (its float32 runs pass, but its mixed-precision runs fail on 1 batch in 8) |
| $\kappa > 10^{6}$ | float64 |
| `basis="eigenbasis"` | float64 only |

Measure the conditioning on a few batches of the data *after* the first BiMap
(`torch.linalg.eigvalsh`), which is what the eigendecompositions see.

## float64, float32 and mixed precision

- **float64 by default.** The conditioning of real covariances reaches
  $10^{6}$ (HDM05) to $10^{7}$ (HyperLeaf); eigendecompositions and the
  divided differences lose too many digits in float32. Every function and layer
  takes `dtype` and `device`.
- **Mixed precision: `eigh_dtype`.** `ReEig`, `ReEigBias`, `LogEig`, the batch
  normalizations, `SpectralVectorField`, `ResidualBlock` and the models take
  `eigh_dtype=torch.float64`: in a float32 model the eigendecompositions are
  computed on the float32 matrices cast to float64 and cast back
  (`spd_linalg.eigh_precision`, `symmetric_eigh`), everything else stays
  float32. float32 `eigh` resolves the eigenvalues only to about
  $\varepsilon_{32} \lVert X \rVert$; in float64 the small eigenvalues of the
  (rounded) float32 matrix are exact to $\varepsilon_{64}\lVert X\rVert$. It
  cannot remove the rounding of the input itself, so it helps most where the
  decomposition, not the data, is the limit (log-Euclidean, ReEig, the
  residual blocks, SPDnet). Against float64 on the original (unrounded) data
  it cannot go below the rounding of the input, about $\kappa\,
  \varepsilon_{32}$ (log-Euclidean batch norm at $\kappa = 10^{6}$: output
  error $7 \cdot 10^{-3}$ in float32, $7 \cdot 10^{-4}$ mixed). On CPU (3
  threads, batch 32, $n = 64$) a mixed forward + backward costs 77–93 %
  (median 86 %) of a float64 one, plain float32 59–81 % (median 72 %). GPU
  timings were not measured.

```python
import torch

from yetanotherspdnet import SPDnet
from yetanotherspdnet.random.spd import random_SPD

model = SPDnet(
    input_dim=32,
    hidden_layers_size=[16],
    output_dim=4,
    dtype=torch.float32,
    eigh_dtype=torch.float64,  # decompositions in float64, the rest in float32
)
X = random_SPD(n_features=32, n_matrices=8, cond=1e4, dtype=torch.float32)
print(model(X).dtype)  # torch.float32
```

## When is float32 enough?

Measured with the precision benchmark (random SPD batches of condition number
$\kappa$, $n \in \{16, 64\}$, batch 32, 8 seeds, CPU; forward + backward in
float32 against float64 on the same float32-rounded input, default options,
training mode). The output error is the larger of the errors against float64
on the rounded input and on the original input. **safe**: no failure, relative
error $\le 10^{-4}$ on the output and $\le 10^{-2}$ on the input gradient; **degraded**: no failure,
output error $\le 10^{-2}$, gradient error $\le 1$; **unsafe**: failures
(exceptions, non-finite values) or larger errors. "mixed" is float32 with
`eigh_dtype=torch.float64`.

| layer | float32: κ ≤ 1e2 | 1e4 | 1e6 | 1e8 | mixed: κ ≤ 1e2 | 1e4 | 1e6 | 1e8 |
|---|---|---|---|---|---|---|---|---|
| ReEig(1e-4) | safe | safe | safe | degraded | safe | safe | safe | safe |
| BN scalar var. [adaptive GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [AI] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [AI (GEVD)] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [arithmetic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [BW] | safe | unsafe | unsafe | unsafe | safe | unsafe | unsafe | unsafe |
| BN scalar var. [GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [harmonic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [LE] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [adaptive GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [AI] | safe | safe | unsafe | unsafe | safe | safe | degraded | unsafe |
| BN mean [AI (GEVD)] | safe | safe | unsafe | unsafe | safe | safe | degraded | unsafe |
| BN mean [arithmetic] | safe | safe | safe | safe | safe | safe | safe | safe |
| BN mean [BW] | safe | safe | degraded | unsafe | safe | safe | degraded | degraded |
| BN mean [GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [harmonic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [LE] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| ResidualBlock [AI] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [LE] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [LE, log output] | safe | degraded | unsafe | unsafe | safe | safe | degraded | unsafe |
| ResidualBlock [congruence] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [branch_pt] | safe | unsafe | unsafe | unsafe | safe | unsafe | unsafe | unsafe |
| ResidualBlock [eigenbasis] | degraded | degraded | degraded | degraded | degraded | degraded | degraded | degraded |
| GBWBNRResNet | safe | safe | safe | unsafe | safe | safe | unsafe | unsafe |
| RResNet | safe | safe | degraded | degraded | safe | safe | safe | degraded |
| SPDnet | safe | safe | safe | degraded | safe | safe | safe | degraded |

The Bures–Wasserstein rows were measured after the SVD cross term
({doc}`manual_backward`), which removed the failures of the
cross term itself: the BW operations (mean, geodesic, centring, std) no
longer fail in float32 up to $\kappa = 10^{6}$ (the geodesic through the
eigendecomposition of $B^{1/2}XB^{1/2}$ failed at $10^{6}$ and was 50 times
less accurate at $10^{4}$), and their forward errors are equal or smaller
everywhere.
The scalar-variance normalization (GBWBN) was unsafe in float32 and mixed
precision from $\kappa = 10^{4}$ at $n = 64$ (12–62 % of the batches): the
eigenvalues of these batches spread over $[10^{-2}, 10^{2}]$, the centring
folds (see {doc}`../user_guide/batchnorm`) and returns matrices that are singular to
rounding, whose square root in the next step has no float32 digits left.
The layer now computes in float64 whatever the dtype of its data
(`yetanotherspdnet.nn.batchnorm.BW_COMPUTE_DTYPE`): the data, $M$, $G$, $s$
and the running statistics are cast on the fly (parameters and buffers keep
their dtype, float32 parameters get float32 gradients) and the output is cast
back, so in a float32 model it matches the float64 layer up to the rounding of
its input and output (`tests/test_float32.py` at $\kappa \le 10^{4}$ with the
tolerances of the other batch normalizations). Its rows in the table above were
measured before that change. Folding itself is not a precision problem: keep
the spectrum close to 1 ($\theta < 1$, `bw_init_metric="trace_scale"` for the
common scale). `GBWBNRResNet` (a BiMap
to $n/2$ first) is safe in float32 up to $\kappa = 10^{6}$; in mixed precision
it still fails on 1 batch in 8 at $\kappa = 10^{6}$ and $n = 64$ (the rows
of the models were measured on the code with both the SVD cross term and the
Cholesky-free residual blocks; the cause of these remaining failures was not
isolated).

The residual blocks do not fail in float32 at $\kappa = 10^{8}$ (12 % of the
runs failed with the previous implementation, through the Cholesky
factorization of the norm and the decomposition of the re-formed floored
matrix). Their log output
(`le_output="log"`) is compared in log coordinates, where the small
eigenvalues weigh as much as the large ones: it carries the same information
as the log-Euclidean block followed by LogEig.

## How it is tested

- The manual backwards are checked with `torch.autograd.gradcheck`
  in float64 (repeated, nearly repeated eigenvalues, the identity,
  $\kappa = 10^{6}$) and compared with the autograd path; every layer is run in
  float32 against a float64 reference for $\kappa \le 10^{4}$, except the
  layer rated unsafe there (`residual="branch_pt"`), tested at
  $\kappa \le 10^{2}$ (`MAX_KAPPA` in `tests/test_float32.py`).

The benchmark protocol is described in {doc}`../development/benchmarks`.

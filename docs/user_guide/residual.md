# Riemannian residual networks

`RResNet` and `GBWBNRResNet` interleave BiMap layers with **residual blocks**
(`nn.rresnet_layers.ResidualBlock`), following Katsman et al., *Riemannian
Residual Neural Networks*, NeurIPS 2023 ([arXiv:2310.10013](https://arxiv.org/abs/2310.10013),
[reference code](https://github.com/CUAI/Riemannian-Residual-Neural-Networks)).

## The residual step

A residual block moves each SPD matrix $X$ along a learned tangent vector,
the Riemannian analogue of $x + f(x)$:

$$
X_{\text{new}} = \mathrm{Exp}_X\big(V(X)\big),
\qquad
V(X) = Q \operatorname{diag}\big(f(\operatorname{spec} X)\big) Q^\top,
$$

where $\operatorname{spec} X$ are the eigenvalues of $X$, $f$ a small Conv1d
network (with BatchNorm1d) and $Q$ a learned orthogonal matrix
(`SpectralVectorField`). The exponential map is chosen with `residual_metric`:

`"affine_invariant"` (default)
: $X_{\text{new}} = \operatorname{projx}\big(\mathrm{Exp}_X(V / \lVert V \rVert_X)\big)$
  with $\lVert V \rVert_X = \lVert X^{-1/2} V X^{-1/2} \rVert_F$: every block
  moves by exactly one unit of affine-invariant distance, and `projx` clamps
  the eigenvalues to $[10^{-8}, 10^{8}]$. This is what the reference code does.

`"log_euclidean"`
: $X_{\text{new}} = \exp\big(\log X + V\big)$, the variant that performs best
  on 3 of the 4 datasets of the paper.

## Numerically singular inputs

The affine-invariant step computes $\lVert V \rVert_X$ with a Cholesky
factorization of $X$. After a batch normalization, $X$ can be numerically
singular: the Bures–Wasserstein steps of GBWBN fold matrices that leave the
injectivity domain of their exponential (see {doc}`batchnorm`), and very
ill-conditioned batches reach the limit of float64 precision. The block
therefore floors the eigenvalues of its input at $10^{-8}$ (the lower bound
of `projx`), with the hand-differentiated ReEig. On well-conditioned inputs
the floor is exactly the identity, values and gradients.

## Why the normalization matters

$\lVert V \rVert_X$ grows like $1 / \lambda_{\min}(X)$. On real covariances,
for instance HyperLeaf with eigenvalues from $10^{-4}$ to $3 \cdot 10^{3}$,
an unnormalized $V$ of Frobenius norm about 3 has an affine-invariant norm
about $2 \cdot 10^{3}$. The matrix exponential then overflows, and the next
eigendecomposition fails. The normalized step, like the log-Euclidean one,
stays finite for any conditioning. `tests/nn/test_rresnet_layers.py`
(`TestResidualBlockIllConditioned`) checks this on matrices spanning that
range.

```python
from yetanotherspdnet import RResNet

model = RResNet(
    input_dim=204,
    hidden_layers_size=[100, 50],
    n_residual_blocks=[1, 1],  # one block after each BiMap
    output_dim=4,
    residual_metric="log_euclidean",
)
```

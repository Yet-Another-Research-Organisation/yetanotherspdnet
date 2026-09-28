# Gradients and precision

## Two gradient paths for every eigen-based operation

Almost every SPD operation goes through an eigendecomposition
$X = U \operatorname{diag}(\lambda) U^\top$ followed by a scalar function on
the eigenvalues, $f(X) = U \operatorname{diag}(f(\lambda)) U^\top$.
Differentiating `torch.linalg.eigh` with autograd involves the terms
$1/(\lambda_i - \lambda_j)$, which blow up when two eigenvalues get close.
This is common after ReEig, which clamps several eigenvalues to the same
threshold.

The library therefore implements each such operation twice:

- a `snake_case` function (`logm_SPD`, `sqrtm_SPD`, `congruence_SPD`, …),
  differentiated by autograd;
- a `CamelCase` `torch.autograd.Function` (`LogmSPD`, `SqrtmSPD`,
  `CongruenceSPD`, …) with a hand-written backward based on the
  Daleckii–Krein formula, which handles equal eigenvalues through the
  derivative $f'(\lambda_i)$ on the diagonal.

Layers and models select the path with `use_autograd` (default `False`,
meaning the manual backward). The models also accept a dict to choose per layer
type:

```python
from yetanotherspdnet import SPDnet

model = SPDnet(
    input_dim=8,
    hidden_layers_size=[4],
    output_dim=3,
    use_autograd={"bimap": True, "batchnorm": True},  # others stay False
)
```

The valid keys are `"bimap"`, `"reeig"`, `"logeig"`, `"batchnorm"` and
`"vec"`; the residual models add `"residual"`. The test suite checks both
paths with `torch.autograd.gradcheck` and checks that their gradients agree.

## Precision: use float64

Eigendecompositions of ill-conditioned matrices lose accuracy quickly in
float32, and gradient checks need float64. All layers and models therefore
default to `dtype=torch.float64`. `random_SPD` defaults to float32, so pass
`dtype=torch.float64` when generating test data.

Every layer and function takes and propagates `device` and `dtype`, so models
run unchanged on GPU (`device=torch.device("cuda")`).

## Parametrizations instead of Riemannian optimizers

Constraints are enforced by `torch.nn.utils.parametrize`, so standard
optimizers (SGD, Adam) can be used:

- BiMap weights live on the Stiefel manifold (orthonormal columns): through
  `torch.nn.utils.parametrizations.orthogonal` in the default `"static"` mode,
  or through `StiefelAdaptiveParametrization` (QR, polar or tangent
  projection around a moving reference point) in the `"dynamic"` mode;
- SPD biases of batch normalization go through a softplus/exp map on the
  eigenvalues (`SPDParametrization`);
- positive scalars go through a softplus (`ScalarSoftPlusParametrization`).

The `"dynamic"` parametrization mode periodically moves the reference point
of the parametrization to the current value. This keeps the optimization in a
well-conditioned chart over long trainings.

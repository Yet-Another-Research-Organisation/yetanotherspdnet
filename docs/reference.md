# API reference

The reference is generated from the docstrings. Each module page opens with a
summary table of its classes and functions. Classes list their attributes
(parameters and buffers), followed by their constructor arguments.

## Top level

| Object | Description |
|---|---|
| {py:class}`~yetanotherspdnet.model.SPDnet` | SPDNet classifier: stacked BiMap/ReEig, optional batch normalization, LogEig and a linear head |
| {py:class}`~yetanotherspdnet.model.RResNet` | Riemannian residual network: BiMap layers interleaved with spectral residual blocks |
| {py:class}`~yetanotherspdnet.model.GBWBNRResNet` | Single-block residual network with GBWBN batch normalization |

## Sub-packages

{py:mod}`yetanotherspdnet.functions.spd_linalg`
: Matrix functions through eigendecomposition (`logm_SPD`, `expm_symmetric`,
  `sqrtm_SPD`, `inv_sqrtm_SPD`, `powm_SPD`, `eigh_relu`), congruences and
  whitening, the Daleckii–Krein gradient `eigh_operation_grad`, Sylvester
  solver, and `vec`/`vech` vectorizations. Each operation comes as a
  `snake_case` function and a `CamelCase` autograd `Function` (see
  {doc}`user_guide/numerics`).

{py:mod}`yetanotherspdnet.functions.spd_geometries`
: One module per geometry, with geodesics, means, dispersions and exp/log
  maps: {py:mod}`~yetanotherspdnet.functions.spd_geometries.affine_invariant`,
  {py:mod}`~yetanotherspdnet.functions.spd_geometries.log_euclidean`,
  {py:mod}`~yetanotherspdnet.functions.spd_geometries.kullback_leibler`,
  {py:mod}`~yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized`,
  {py:mod}`~yetanotherspdnet.functions.spd_geometries.bures_wasserstein`
  (see {doc}`user_guide/geometries`).

{py:mod}`yetanotherspdnet.functions.m_estimators`
: Sample covariance and robust M-estimators of scatter (Tyler, Student-t,
  Huber weights), with an unrolled autograd path and an implicit
  fixed-point backward.

{py:mod}`yetanotherspdnet.functions.scalar_functions`, {py:mod}`yetanotherspdnet.functions.stiefel`
: Scalar maps applied to eigenvalues (and their derivatives), and
  projections/retractions on the Stiefel manifold.

{py:mod}`yetanotherspdnet.nn.base`
: The SPDNet layers: `BiMap`, `ReEig`, `ReEigBias` (learned eigenvalue
  shift, two-sided clamp), `LogEig`, `Vec`, `Vech`.

{py:mod}`yetanotherspdnet.nn.estimation`
: `SampleCovariance` and `MEstimation`: layers mapping raw samples to SPD
  matrices, differentiable with respect to the samples.

{py:mod}`yetanotherspdnet.nn.batchnorm`
: `BatchNormSPDMean` and `BatchNormSPDMeanScalarVariance`
  (see {doc}`user_guide/batchnorm`).

{py:mod}`yetanotherspdnet.nn.rresnet_layers`
: `SpectralVectorField` and `ResidualBlock`, the building blocks of the
  residual networks.

{py:mod}`yetanotherspdnet.nn.parametrizations`
: SPD, Stiefel and positive-scalar parametrizations used to keep parameters
  on their manifold.

{py:mod}`yetanotherspdnet.random`
: `random_SPD` (with a prescribed condition number) and random Stiefel
  matrices.

## Full index

```{toctree}
:maxdepth: 2

reference/yetanotherspdnet/index
```

# Quickstart

Every snippet on this page runs as-is against the current version.

```{important}
Work in **float64**. All layers and models default to `torch.float64`, because
eigendecompositions lose accuracy quickly in float32. `random_SPD` defaults to
float32, so pass `dtype=torch.float64` explicitly (mixing both raises a dtype
error in the first layer).
```

## SPD matrices and matrix functions

```python
import torch
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.functions.spd_linalg import expm_symmetric, logm_SPD

generator = torch.Generator().manual_seed(0)
X = random_SPD(
    n_features=8, n_matrices=32, cond=100, dtype=torch.float64, generator=generator
)
print(X.shape)  # torch.Size([32, 8, 8]): a batch of 32 matrices 8 x 8

log_X = logm_SPD(X)[0]  # matrix functions return (result, eigvals, eigvecs)
print(torch.allclose(expm_symmetric(log_X)[0], X))  # True
```

Every function accepts any number of leading batch dimensions
`(..., n, n)`: `(B, n, n)` for a batch, `(B, T, n, n)` for sequences.

## Riemannian means

The mean of SPD matrices depends on the geometry. Each geometry lives in its
own module under `functions.spd_geometries`:

```python
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_mean,
)
from yetanotherspdnet.functions.spd_geometries.bures_wasserstein import (
    bures_wasserstein_mean,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import log_euclidean_mean

G_ai = affine_invariant_mean(X, n_iterations=10)  # Karcher flow
G_le = log_euclidean_mean(X)  # closed form
G_bw = bures_wasserstein_mean(X, n_iterations=10)  # fixed point
print(G_ai.shape)  # torch.Size([8, 8])
```

See {doc}`user_guide/geometries` for the formulas and when to use which.

## Layers

Layers are regular `torch.nn.Module`s and compose with `torch.nn.Sequential`:

```python
from yetanotherspdnet.nn import BiMap, LogEig, ReEig, Vech

features = torch.nn.Sequential(
    BiMap(8, 4),  # 8x8 -> 4x4, orthonormal weight
    ReEig(eps=1e-4),  # clamp eigenvalues below eps
    LogEig(),  # SPD -> symmetric
    Vech(),  # 4x4 symmetric -> 10 coefficients
)
print(features(X).shape)  # torch.Size([32, 10])
```

## A complete model

```python
from yetanotherspdnet import SPDnet

model = SPDnet(
    input_dim=8,
    hidden_layers_size=[6, 4],  # BiMap 8->6->4
    output_dim=3,  # number of classes
    batchnorm=True,
    batchnorm_mean_type="affine_invariant",
)
print(model(X).shape)  # torch.Size([32, 3])
```

It trains like any PyTorch model:

```python
y = torch.randint(0, 3, (32,), generator=generator)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

model.eval()  # batch normalization now uses its running statistics
with torch.no_grad():
    predictions = model(X).argmax(dim=-1)
```

The orthonormality of the BiMap weights and the positivity of the batch
normalization biases are handled by `torch.nn.utils.parametrize`, so a plain
Euclidean optimizer such as Adam or SGD is enough.

## Batch normalization on its own

```python
from yetanotherspdnet.nn import BatchNormSPDMeanScalarVariance

bn = BatchNormSPDMeanScalarVariance(n_features=8, mean_type="log_euclidean")
print(bn(X).shape)  # torch.Size([32, 8, 8])
```

See {doc}`user_guide/batchnorm` for the available geometries and options.

## Residual networks

```python
from yetanotherspdnet import GBWBNRResNet, RResNet

rresnet = RResNet(
    input_dim=8, hidden_layers_size=[6, 4], n_residual_blocks=[1, 1], output_dim=3
)
gbwbn = GBWBNRResNet(input_dim=8, hidden_dim=4, output_dim=3)
print(rresnet(X).shape, gbwbn(X).shape)  # torch.Size([32, 3]) torch.Size([32, 3])
```

## Where next

| To go further on… | Read |
|---|---|
| what happens inside `SPDnet`, and the shape at each layer | {doc}`user_guide/concepts` |
| the constructor arguments, grouped by role, and published configurations | {doc}`user_guide/models` |
| BiMap's `parametrization_mode="dynamic"` (needs `model.register_optimizer_hook(optimizer)`) | {doc}`user_guide/layers` |
| the batch normalization options (geometry, running statistics, GBWBN) | {doc}`user_guide/batchnorm` |
| why the gradients stay finite on ill-conditioned data | {doc}`user_guide/numerics` |

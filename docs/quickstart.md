# Quick Start Guide

This guide will help you get started with Yet Another SPDNet.

## Basic SPD Matrix Operations

### Creating Random SPD Matrices

```python
import torch
from yetanotherspdnet import spd

# Generate random SPD matrices
n_features = 50
n_matrices = 10
X = spd.random_SPD(n_features, n_matrices, cond=100)

print(f"Shape: {X.shape}")  # (10, 50, 50)
```

### Matrix Logarithm and Exponential

```python
# Compute matrix logarithm
X_log = spd.logm_SPD(X)

# Compute matrix exponential (inverse operation)
X_reconstructed = spd.expm_symmetric(X_log)

# Verify reconstruction
print(torch.allclose(X, X_reconstructed))  # True
```

### Square Root and Inverse Square Root

```python
# Compute matrix square root
X_sqrt = spd.sqrtm_SPD(X)

# Verify: X_sqrt @ X_sqrt = X
print(torch.allclose(X_sqrt @ X_sqrt, X))  # True

# Compute inverse square root
X_inv_sqrt = spd.inv_sqrtm_SPD(X)
```

## Computing SPD Means

```python
from yetanotherspdnet import means

# Arithmetic mean (Euclidean)
mean_arith = means.mean_arithmetic(X)

# Geometric mean (Riemannian)
mean_geom = means.mean_geometric(X, max_iter=10)

# Log-Euclidean mean
mean_log = means.mean_log_euclidean(X)

# Harmonic mean
mean_harm = means.mean_harmonic(X)

print(f"Arithmetic mean shape: {mean_arith.shape}")  # (50, 50)
```

## Building an SPDNet Model

```python
from yetanotherspdnet.models import SPDNet

# Create an SPDNet model
model = SPDNet(
    input_dim=50,
    hidden_layers_size=[30, 20],
    output_dim=10,
    softmax=True,
    batchnorm=True,
    batchnorm_method="geometric",
    eps=1e-3,
)

# Forward pass
X = spd.random_SPD(50, batch_size=32)
output = model(X)

print(f"Output shape: {output.shape}")  # (32, 10)
```

## Using Neural Network Layers

```python
from yetanotherspdnet.nn import BiMap, ReEig, LogEig, Vec

# BiMap layer (bilinear mapping)
bimap = BiMap(n_in=50, n_out=30)
X_mapped = bimap(X)
print(f"After BiMap: {X_mapped.shape}")  # (32, 30, 30)

# ReEig layer (eigenvalue rectification)
reeig = ReEig(eps=1e-3, dim=30)
X_rectified = reeig(X_mapped)

# LogEig layer (matrix logarithm)
logeig = LogEig()
X_log = logeig(X_rectified)

# Vec layer (vectorization)
vec = Vec()
X_vec = vec(X_log)
print(f"After Vec: {X_vec.shape}")  # (32, 900)
```

## Training Example

```python
import torch.nn as nn
import torch.optim as optim

# Create model
model = SPDNet(
    input_dim=50,
    hidden_layers_size=[30, 20],
    output_dim=5,
    softmax=True,
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    # Generate batch of SPD matrices and labels
    X_batch = spd.random_SPD(50, batch_size=32)
    y_batch = torch.randint(0, 5, (32,))

    # Forward pass
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Batch Normalization for SPD Matrices

```python
from yetanotherspdnet.nn import BatchNormSPDMean

# Create batch normalization layer
batchnorm = BatchNormSPDMean(
    n_features=50,
    mean_type="geometric",
    momentum=0.1,
)

# Apply batch normalization
X_normalized = batchnorm(X)
```

## Next Steps

- Explore the [API Reference](api.md) for detailed documentation
- Check out the [Contributing Guide](contributing.md) to contribute
- Read the source code for advanced usage patterns

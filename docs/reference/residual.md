# Residual blocks

`yetanotherspdnet.nn.rresnet_layers` — the building blocks of `RResNet` and
`GBWBNRResNet`. `SpectralVectorField` computes a tangent vector from the
spectrum of each matrix; `ResidualBlock` moves the matrix along it with an
affine-invariant (unit step) or log-Euclidean step, from two
eigendecompositions (`FlooredEighFunctions` in
{doc}`spd_linalg`). See {doc}`../user_guide/residual`.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.nn.rresnet_layers

.. autosummary::
   :nosignatures:

   ResidualBlock
   SpectralVectorField
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.rresnet_layers.ResidualBlock
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.rresnet_layers.SpectralVectorField
   :members: forward, spectrum, from_spectrum, register_optimizer_hook
```

The affine-invariant norm used to normalize the step,
{func}`~yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_norm`,
is documented with the affine-invariant geometry.

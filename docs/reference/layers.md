# Layers

`yetanotherspdnet.nn` — the SPDNet layers. `BiMap` changes the dimension,
`ReEig` / `ReEigBias` are the non-linearities, `LogEig` and `Vec` / `Vech` map
to a vector for a Euclidean head. `SampleCovariance` and `MEstimation` build
SPD matrices from raw samples, differentiably. See {doc}`../user_guide/layers`
for the equations and the static / dynamic parametrization of `BiMap`.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.nn

.. autosummary::
   :nosignatures:

   BiMap
   ReEig
   ReEigBias
   LogEig
   Vec
   Vech
   SampleCovariance
   MEstimation
```

## SPDNet layers

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.BiMap
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.ReEig
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.ReEigBias
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.LogEig
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.Vec
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.Vech
   :members: forward
```

## Covariance estimation

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.SampleCovariance
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.MEstimation
   :members: forward
```

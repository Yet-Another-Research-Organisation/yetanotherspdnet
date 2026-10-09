# Batch normalization

`yetanotherspdnet.nn.batchnorm` — centring (and optionally rescaling) a batch
of SPD matrices around a learned bias, in any of the {doc}`geometries`. The
{doc}`../user_guide/batchnorm` guide describes the steps, the choice of mean,
the training / evaluation behaviour and the GBWBN options.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.nn.batchnorm

.. autosummary::
   :nosignatures:

   BatchNormSPDMean
   BatchNormSPDMeanScalarVariance
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.batchnorm.BatchNormSPDMean
   :members: forward
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.batchnorm.BatchNormSPDMeanScalarVariance
   :members: forward
```

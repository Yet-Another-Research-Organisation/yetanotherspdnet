# Covariance estimators

`yetanotherspdnet.functions.m_estimators` — from samples $x_1, \dots, x_n \in
\mathbb{R}^p$ to a scatter matrix. The sample covariance is
$\frac1n \sum_i x_i x_i^\top$; an M-estimator solves the fixed point

$$
\Sigma = \frac1n \sum_{i=1}^n u\!\left(x_i^\top \Sigma^{-1} x_i\right) x_i x_i^\top,
$$

whose weight $u$ down-weights outlying samples (Tyler: $u(q) = p/q$;
Student-t: $u(q) = (p+\nu)/(\nu+q)$; Huber). `m_estimator` differentiates the
unrolled iterations; `MEstimator` differentiates the fixed point implicitly,
with a memory cost independent of the number of iterations. The layers
`SampleCovariance` and `MEstimation` ({doc}`layers`) wrap them.

```{dualpath-table} yetanotherspdnet.functions.m_estimators
```

## Details

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.m_estimators
   :members:
   :member-order: bysource
```

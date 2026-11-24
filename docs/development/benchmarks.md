# Numerical benchmarks

The float32 / mixed-precision table of {doc}`../numerics/precision` and the
timings quoted in the user guide come from a precision benchmark run on CPU.
Its protocol, so that it can be rerun and extended:

## Precision

- **Inputs.** Random SPD batches (`random_SPD`) of condition number
  $\kappa \in \{10, 10^{2}, 10^{4}, 10^{6}, 10^{8}\}$, $n \in \{16, 64\}$,
  batch 32, 8 seeds.
- **What is run.** Each layer (ReEig, both batch normalizations in every
  geometry, the residual blocks and their options) and each model, with its
  default options, in training mode: forward and backward (a fixed random
  cotangent).
- **Reference.** The same layer in float64, with the same parameters, on the
  same float32-rounded input; the output error is also measured against
  float64 on the original input, and the larger of the two is kept.
- **Variants.** float32 (manual backward), float32 with
  `eigh_dtype=torch.float64` ("mixed").
- **Metrics.** Maximal relative error of the output and of the input
  gradient; failure rate (exceptions or non-finite values).
- **Verdict.** *safe*: no failure, output error $\le 10^{-4}$ and gradient
  error $\le 10^{-2}$; *degraded*: no failure, output error $\le 10^{-2}$,
  gradient error $\le 1$; *unsafe*: anything else. $\kappa \le 10^{2}$ pools
  $\kappa = 10$ and $10^{2}$.

`tests/test_float32.py` turns the measured errors into tolerances (about ten
times the largest error at $\kappa \le 10^{4}$), so a regression in float32
fails the test suite.

## Timings

Forward + backward wall time, median of repeated runs, CPU with 3 threads,
batch 32 or 64, $n \in \{16, 64\}$, float32, mixed and float64. GPU timings
were not measured. The residual-block timings compare the fused block (two
eigendecompositions) with the previous implementation (five and a Cholesky).

## Rerunning

The benchmark scripts are kept outside the repository. To rerun the
comparison on a change, apply the protocol above to the layers you touch, on
the branch and on `main`, with the same seeds and thread count, and report the
before / after tables in the pull request.

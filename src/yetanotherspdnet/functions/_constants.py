"""Numerical constants shared by the library."""

# Daleckii-Krein divided differences (f(a) - f(b)) / (a - b), see
# spd_linalg._aux_eigh_operation_grad.
# Closed-form kernels (sqrt, inverse sqrt, log, exp, power, inverse) do not cancel
# for any gap a - b != 0: they only need to catch ties at rounding level,
# |a - b| <= EIGVAL_TIE_RTOL * n * eps(dtype) * max|lambda| (per matrix), where the
# derivative at (a + b) / 2 is used instead.
EIGVAL_TIE_RTOL = 1.0
# Generic quotient (f(a) - f(b)) / (a - b): its rounding error ~ eps |a| / |a - b|
# grows as the gap shrinks, while the error of f'((a + b) / 2) ~ (|a - b| / |a|)^2 / 24
# shrinks. Both balance at |a - b| = (24 eps)^(1/3) max(|a|, |b|), the switch used
# for functions without a closed-form kernel (softplus, ReEig clamp, ...);
# the error there is ~ eps^(2/3) (5e-11 in float64, 2e-5 in float32).
EIGVAL_TIE_GENERIC_FACTOR = 24.0

# step size eta_k = KARCHER_STEP_DECAY ** k of the k-th iteration of the
# affine-invariant Karcher mean (fixed-point) algorithm
KARCHER_STEP_DECAY = 0.95

# eigenvalue bounds of affine_invariant_projx, as in the reference RResNet
# implementation; the lower one is also the input floor of ResidualBlock
SPD_EIGVAL_FLOOR = 1e-8
SPD_EIGVAL_CEILING = 1e8

# spd_linalg.spd_floor: eigenvalues below SPD_FLOOR_RTOL * n * eps(dtype) * max|lambda|
# are below the rounding noise of the decomposition (backward error ~ n eps ||X||)
SPD_FLOOR_RTOL = 10.0

# lower bound of the affine-invariant norm of the residual step (ResidualBlock)
TANGENT_NORM_FLOOR = 1e-12

# clamp of t to [eps, 1 - eps] before the logit of ScalarSigmoidParametrization
SIGMOID_CLAMP_EPS = 1e-7

# variance regularization of the GBWBN scaling s / sqrt(var + BW_SCALE_EPS)
BW_SCALE_EPS = 1e-5
# lower bound of the scalar std in the scaling exponent s / std of
# BatchNormSPDMeanScalarVariance: bounds it by s / sqrt(BW_SCALE_EPS) for a batch of
# identical matrices (std = 0), as the GBWBN scaling; no effect for
# std >= 3.2e-3
STD_FLOOR = BW_SCALE_EPS**0.5

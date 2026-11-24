"""Numerical constants shared by the library."""

# Daleckii-Krein divided differences (f(a) - f(b)) / (a - b): below this absolute
# gap between two eigenvalues, the derivative f'(a) is used instead
EIGVAL_TIE_THRESHOLD = 1e-6

# step size eta_k = KARCHER_STEP_DECAY ** k of the k-th iteration of the
# affine-invariant Karcher mean (fixed-point) algorithm
KARCHER_STEP_DECAY = 0.95

# eigenvalue bounds of affine_invariant_projx, as in the reference RResNet
# implementation; the lower one is also the input floor of ResidualBlock
SPD_EIGVAL_FLOOR = 1e-8
SPD_EIGVAL_CEILING = 1e8

# lower bound of the affine-invariant norm of the residual step (ResidualBlock)
TANGENT_NORM_FLOOR = 1e-12

# clamp of t to [eps, 1 - eps] before the logit of ScalarSigmoidParametrization
SIGMOID_CLAMP_EPS = 1e-7

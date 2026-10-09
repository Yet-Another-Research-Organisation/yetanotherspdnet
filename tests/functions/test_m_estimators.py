from functools import partial

import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.m_estimators as me
from utils import is_spd
from yetanotherspdnet.nn import MEstimation, SampleCovariance


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="function")
def generator(device):
    generator = torch.Generator(device=device)
    generator.manual_seed(777)
    return generator


def _samples(n_batches, n_samples, n_features, device, dtype, generator, df=None):
    """Correlated Gaussian (or Student-t with ``df`` degrees of freedom) samples."""
    mixing = torch.randn(
        n_features, n_features, device=device, dtype=dtype, generator=generator
    ) + 2 * torch.eye(n_features, device=device, dtype=dtype)
    data = torch.randn(
        n_batches,
        n_samples,
        n_features,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if df is not None:
        chi2 = (
            torch.randn(
                n_batches,
                n_samples,
                df,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            ** 2
        ).sum(dim=-1, keepdim=True)
        data = data * torch.sqrt(df / chi2)
    return data @ mixing.T


class TestWeightsAndNormalization:
    def test_huber_weights(self, device, dtype):
        quadratic = torch.tensor([0.5, 1.0, 4.0], device=device, dtype=dtype)
        weights = me.huber_function(quadratic, delta=1.0, beta=2.0)
        expected = torch.tensor([0.5, 0.5, 0.125], device=device, dtype=dtype)
        assert_close(weights, expected)

    def test_student_tends_to_tyler(self, device, dtype):
        quadratic = torch.tensor([0.5, 3.0], device=device, dtype=dtype)
        assert_close(
            me.student_function(quadratic, n_features=4, nu=1e-10),
            me.tyler_function(quadratic, n_features=4),
        )

    @pytest.mark.parametrize("normalize", ["trace", "determinant"])
    def test_normalizations(self, normalize, device, dtype, generator):
        data = _samples(3, 20, 4, device, dtype, generator)
        covariance = me.sample_covariance(data) * 1e3
        normalized = me._get_normalization(normalize)(covariance)
        if normalize == "trace":
            trace = torch.diagonal(normalized, dim1=-2, dim2=-1).sum(-1)
            assert_close(trace, torch.full_like(trace, 4.0))
        else:
            assert_close(
                torch.linalg.det(normalized),
                torch.ones(3, device=device, dtype=dtype),
            )

    def test_unknown_normalization(self):
        with pytest.raises(ValueError):
            me._get_normalization("frobenius")


class TestSampleCovariance:
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_matches_torch_cov(self, assume_centered, device, dtype, generator):
        data = _samples(2, 30, 5, device, dtype, generator)
        covariance = SampleCovariance(assume_centered)(data)
        for b in range(2):
            if assume_centered:
                expected = data[b].T @ data[b] / 30
            else:
                expected = torch.cov(data[b].T)
            assert_close(covariance[b], expected)
        assert is_spd(covariance)


class TestMEstimator:
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_tyler_is_fixed_point(self, use_autograd, device, dtype, generator):
        data = _samples(3, 50, 4, device, dtype, generator, df=2)
        covariance = me.tyler_estimator(
            data, n_iterations=300, tol=1e-12, use_autograd=use_autograd
        )
        centered = data - data.mean(dim=-2, keepdim=True)
        step = me.m_estimator_step(
            covariance,
            centered,
            partial(me.tyler_function, n_features=4),
            normalize=me.normalize_trace,
        )
        assert_close(step, covariance, atol=1e-9, rtol=0)
        assert is_spd(covariance)

    def test_tyler_scale_invariance(self, device, dtype, generator):
        data = _samples(2, 50, 4, device, dtype, generator)
        estimate = me.tyler_estimator(data, n_iterations=300, tol=1e-12)
        estimate_scaled = me.tyler_estimator(10 * data, n_iterations=300, tol=1e-12)
        assert_close(estimate, estimate_scaled, atol=1e-8, rtol=0)

    def test_student_large_nu_is_sample_covariance(self, device, dtype, generator):
        data = _samples(2, 60, 4, device, dtype, generator)
        estimate = me.student_estimator(data, nu=1e9, n_iterations=100, tol=1e-12)
        expected = me.sample_covariance(data) * 59 / 60  # 1/n normalization
        assert_close(estimate, expected, atol=1e-6, rtol=0)

    def test_student_is_robust_to_outliers(self, device, dtype, generator):
        data = _samples(1, 200, 3, device, dtype, generator)
        clean_scm = me.sample_covariance(data)
        clean_student = me.student_estimator(data, nu=3.0, n_iterations=200)
        corrupted = data.clone()
        corrupted[:, :5] *= 100.0
        scm_shift = (me.sample_covariance(corrupted) - clean_scm).norm()
        student_shift = (
            me.student_estimator(corrupted, nu=3.0, n_iterations=200) - clean_student
        ).norm()
        assert student_shift < 0.1 * scm_shift

    def test_shrinkage_regularizes(self, device, dtype, generator):
        # fewer samples than features: the unregularized estimate is singular
        data = _samples(1, 4, 6, device, dtype, generator)
        weight = partial(me.student_function, n_features=6, nu=3.0)
        estimate = me.m_estimator(data, weight, n_iterations=100, shrinkage=0.7)
        assert is_spd(estimate)

    def test_tyler_requires_normalization(self, device, dtype, generator):
        data = _samples(1, 20, 3, device, dtype, generator)
        with pytest.raises(ValueError):
            me.tyler_estimator(data, normalize=None)

    @pytest.mark.parametrize("estimator", ["tyler", "student"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_gradcheck(self, estimator, use_autograd, device, dtype, generator):
        data = _samples(1, 10, 3, device, dtype, generator).requires_grad_(True)
        if estimator == "tyler":
            fun = lambda x: me.tyler_estimator(  # noqa: E731
                x, n_iterations=300, tol=1e-13, use_autograd=use_autograd
            )
        else:
            fun = lambda x: me.student_estimator(  # noqa: E731
                x, nu=3.0, n_iterations=300, tol=1e-13, use_autograd=use_autograd
            )
        assert torch.autograd.gradcheck(fun, (data,), atol=1e-6)

    @pytest.mark.parametrize("normalize", [None, "trace"])
    def test_gradient_paths_agree(self, normalize, device, dtype, generator):
        data = _samples(3, 40, 4, device, dtype, generator)
        weight = partial(me.student_function, n_features=4, nu=4.0)
        grad_output = torch.randn(
            3, 4, 4, device=device, dtype=dtype, generator=generator
        )
        grads = []
        for use_autograd in (True, False):
            leaf = data.clone().requires_grad_(True)
            layer = MEstimation(
                weight,
                n_iterations=300,
                tol=1e-13,
                normalize=normalize,
                use_autograd=use_autograd,
            )
            (layer(leaf) * grad_output).sum().backward()
            grads.append(leaf.grad)
        assert_close(grads[0], grads[1], atol=1e-9, rtol=0)

    def test_float32_manual_path(self, device, generator):
        data = _samples(2, 40, 4, device, torch.float64, generator).float()
        data.requires_grad_(True)
        covariance = me.tyler_estimator(data, n_iterations=100, tol=1e-6)
        covariance.sum().backward()
        assert covariance.dtype == torch.float32
        assert torch.isfinite(data.grad).all()

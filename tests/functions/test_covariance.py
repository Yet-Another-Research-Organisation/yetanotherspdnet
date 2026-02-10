import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.covariance as covariance
from utils import is_spd, is_symmetric
from yetanotherspdnet.functions.spd_linalg import sqrtm_SPD
from yetanotherspdnet.random.spd import random_SPD

from sklearn.covariance import empirical_covariance, ledoit_wolf


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


class TestSampleCovariance:
    """
    Test suite for sample covariance estimator
    """

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_forward_shape(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        Test that output of sample covariance functions is coherent
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )
        # covariance estimation
        SCM_manual = covariance.SampleCovariance.apply(data, assume_centered)
        SCM_auto = covariance.sample_covariance(data, assume_centered)

        if n_matrices > 1:
            assert SCM_manual.shape[0] == n_matrices
        assert SCM_manual.shape[-2] == n_features
        assert SCM_manual.shape[-1] == n_features
        assert is_spd(SCM_manual)
        assert SCM_manual.device == data.device
        assert SCM_manual.dtype == data.dtype

        if n_matrices > 1:
            assert SCM_auto.shape[0] == n_matrices
        assert SCM_auto.shape[-2] == n_features
        assert SCM_auto.shape[-1] == n_features
        assert is_spd(SCM_auto)
        assert SCM_auto.device == data.device
        assert SCM_auto.dtype == data.dtype

        assert_close(SCM_manual, SCM_auto)

    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    # TODO: result is different from sklearn when assume centered is False
    # because they still take n_samples when we take n_samples - 1
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_sklearn_comparison(
        self, n_samples, n_features, cond, assume_centered, device, dtype, generator
    ):
        """
        Test that we get the same result as sklearn function empirical_covariance
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.randn(
                (n_samples, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            @ Cov_sqrtm
        )
        # covariance estimation
        SCM_manual = covariance.SampleCovariance.apply(data, assume_centered)
        SCM_auto = covariance.sample_covariance(data, assume_centered)
        # sklearn estimator
        SCM_sklearn = torch.from_numpy(
            empirical_covariance(data.numpy(), assume_centered=assume_centered)
        ).to(dtype=dtype, device=device)

        assert_close(SCM_manual, SCM_sklearn)
        assert_close(SCM_auto, SCM_sklearn)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_backward(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        test that manual and autograd gradients are the same
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )

        X_manual = data.clone().detach()
        X_manual.requires_grad = True
        X_auto = data.clone().detach()
        X_auto.requires_grad = True

        SCM_manual = covariance.SampleCovariance.apply(X_manual, assume_centered)
        SCM_auto = covariance.sample_covariance(X_auto, assume_centered)

        loss_manual = torch.norm(SCM_manual)
        loss_manual.backward()
        loss_auto = torch.norm(SCM_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)


class TestLedoitWolfCovariance:
    """
    Test suite for Ledoit-Wolf covariance estimator
    """

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_forward_shape(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        Test that output of sample covariance functions is coherent
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )
        # covariance estimation
        LW_manual = covariance.LedoitWolfCovariance.apply(data, assume_centered)
        LW_auto = covariance.ledoit_wolf_covariance(data, assume_centered)

        if n_matrices > 1:
            assert LW_manual.shape[0] == n_matrices
        assert LW_manual.shape[-2] == n_features
        assert LW_manual.shape[-1] == n_features
        assert is_spd(LW_manual)
        assert LW_manual.device == data.device
        assert LW_manual.dtype == data.dtype

        if n_matrices > 1:
            assert LW_auto.shape[0] == n_matrices
        assert LW_auto.shape[-2] == n_features
        assert LW_auto.shape[-1] == n_features
        assert is_spd(LW_auto)
        assert LW_auto.device == data.device
        assert LW_auto.dtype == data.dtype

        assert_close(LW_manual, LW_auto)

    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    # TODO: result is different from sklearn when assume centered is False
    # because they still take n_samples when we take n_samples - 1
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_sklearn_comparison(
        self, n_samples, n_features, cond, assume_centered, device, dtype, generator
    ):
        """
        Test that we get the same result as sklearn function empirical_covariance
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.randn(
                (n_samples, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            @ Cov_sqrtm
        )
        # covariance estimation
        LW_manual = covariance.LedoitWolfCovariance.apply(data, assume_centered)
        LW_auto = covariance.ledoit_wolf_covariance(data, assume_centered)
        # sklearn estimator
        LW_sklearn = torch.from_numpy(
            ledoit_wolf(data.numpy(), assume_centered=assume_centered)[0]
        ).to(dtype=dtype, device=device)

        assert_close(LW_manual, LW_sklearn)
        assert_close(LW_auto, LW_sklearn)

    @pytest.mark.parametrize("n_matrices", [1, 2])
    @pytest.mark.parametrize("n_samples", [10, 20])
    @pytest.mark.parametrize("n_features, cond", [(5, 100)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_backward(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        test that manual and autograd gradients are the same
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )

        X_manual = data.clone().detach()
        X_manual.requires_grad = True
        X_auto = data.clone().detach()
        X_auto.requires_grad = True

        LW_manual = covariance.LedoitWolfCovariance.apply(X_manual, assume_centered)
        LW_auto = covariance.ledoit_wolf_covariance(X_auto, assume_centered)

        loss_manual = torch.norm(LW_manual)
        loss_manual.backward()
        loss_auto = torch.norm(LW_auto)
        loss_auto.backward()

        # print(X_manual.grad)
        # print(X_auto.grad)

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)


class TestSampleVariance:
    """
    Test suite for sample variance vector estimator
    """

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_forward_shape(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        Test that output of sample variance functions is coherent
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )
        # variance estimation
        SV_manual = covariance.SampleVariance.apply(data, assume_centered)
        SV_auto = covariance.sample_variance(data, assume_centered)

        if n_matrices > 1:
            assert SV_manual.shape[0] == n_matrices
        assert SV_manual.shape[-1] == n_features
        assert SV_manual.device == data.device
        assert SV_manual.dtype == data.dtype
        assert (SV_manual > 0).all()

        if n_matrices > 1:
            assert SV_auto.shape[0] == n_matrices
        assert SV_auto.shape[-1] == n_features
        assert SV_auto.device == data.device
        assert SV_auto.dtype == data.dtype
        assert (SV_auto > 0).all()

        assert_close(SV_manual, SV_auto)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_SCM_comparison(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        Test that sample variance is the diagonal of sample covariance
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )
        # variance estimation
        SV_manual = covariance.SampleVariance.apply(data, assume_centered)
        SV_auto = covariance.sample_variance(data, assume_centered)
        # covariance estimation
        SCM = covariance.sample_covariance(data, assume_centered)
        SCM_variance = torch.diagonal(SCM, dim1=-1, dim2=-2)

        assert_close(SV_manual, SCM_variance)
        assert_close(SV_auto, SCM_variance)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_samples", [200, 1000])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("assume_centered", [True, False])
    def test_backward(
        self,
        n_matrices,
        n_samples,
        n_features,
        cond,
        assume_centered,
        device,
        dtype,
        generator,
    ):
        """
        test that manual and autograd gradients are the same
        """
        # random covariance matrices
        Cov = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Cov_sqrtm = sqrtm_SPD(Cov)[0]
        # random data
        data = (
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_samples, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
            @ Cov_sqrtm
        )

        X_manual = data.clone().detach()
        X_manual.requires_grad = True
        X_auto = data.clone().detach()
        X_auto.requires_grad = True

        SV_manual = covariance.SampleVariance.apply(X_manual, assume_centered)
        SV_auto = covariance.sample_variance(X_auto, assume_centered)

        loss_manual = torch.norm(SV_manual)
        loss_manual.backward()
        loss_auto = torch.norm(SV_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

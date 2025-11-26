import pytest
import torch
from torch.testing import assert_close

from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_mean_2points,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import arithmetic_mean
import yetanotherspdnet.functions.spd_linalg as spd_linalg

import yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized as kullback_leibler_symmetrized

from yetanotherspdnet.random.spd import random_SPD, random_DPD
from yetanotherspdnet.random.stiefel import random_stiefel

from utils import is_symmetric, is_spd


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


class TestGeometricArithmeticHarmonicAdaptiveUpdate:
    """
    Test suite for adaptive update of the geometric mean of arithmetic and harmonic means
    """

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_shape(self, n_features, cond, device, dtype, generator):
        """
        Test that geometric_arithmetic_harmonic_adaptive_update function works and return appropriate shape
        """
        points = random_SPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1_eucl = points[0]
        point1_harm = points[1]
        point2_eucl = points[2]
        point2_harm = points[3]

        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto, point_eucl_auto, point_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )
        point_manual, point_eucl_manual, point_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )

        assert point_auto.shape == point1_eucl.shape
        assert point_auto.device == point1_eucl.device
        assert point_auto.dtype == point1_eucl.dtype
        assert is_spd(point_auto)

        assert point_eucl_auto.shape == point1_eucl.shape
        assert point_eucl_auto.device == point1_eucl.device
        assert point_eucl_auto.dtype == point1_eucl.dtype
        assert is_spd(point_eucl_auto)

        assert point_harm_auto.shape == point1_eucl.shape
        assert point_harm_auto.device == point1_eucl.device
        assert point_harm_auto.dtype == point1_eucl.dtype
        assert is_spd(point_harm_auto)

        assert point_manual.shape == point1_eucl.shape
        assert point_manual.device == point1_eucl.device
        assert point_manual.dtype == point1_eucl.dtype
        assert is_spd(point_manual)

        assert point_eucl_manual.shape == point1_eucl.shape
        assert point_eucl_manual.device == point1_eucl.device
        assert point_eucl_manual.dtype == point1_eucl.dtype
        assert is_spd(point_eucl_manual)

        assert point_harm_manual.shape == point1_eucl.shape
        assert point_harm_manual.device == point1_eucl.device
        assert point_harm_manual.dtype == point1_eucl.dtype
        assert is_spd(point_harm_manual)

        assert_close(point_eucl_auto, point_eucl_manual)
        assert_close(point_harm_auto, point_harm_manual)
        assert_close(point_auto, point_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_extremities(self, n_features, cond, device, dtype, generator):
        """
        Test that for t=0 and t=1 we get point1 and point2
        """
        points = random_SPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1_eucl = points[0]
        point1_harm = points[1]
        point2_eucl = points[2]
        point2_harm = points[3]

        point1_auto, point1_eucl_auto, point1_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, 0
            )
        )
        point2_auto, point2_eucl_auto, point2_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, 1
            )
        )
        point1_manual, point1_eucl_manual, point1_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, 0
            )
        )
        point2_manual, point2_eucl_manual, point2_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, 1
            )
        )

        assert_close(point1_eucl_auto, point1_eucl)
        assert_close(point1_harm_auto, point1_harm)
        assert_close(
            point1_auto, affine_invariant_mean_2points(point1_eucl, point1_harm)
        )
        assert_close(point2_eucl_auto, point2_eucl)
        assert_close(point2_harm_auto, point2_harm)
        assert_close(
            point2_auto, affine_invariant_mean_2points(point2_eucl, point2_harm)
        )
        assert_close(point1_eucl_manual, point1_eucl)
        assert_close(point1_harm_manual, point1_harm)
        assert_close(
            point1_manual, affine_invariant_mean_2points(point1_eucl, point1_harm)
        )
        assert_close(point2_eucl_manual, point2_eucl)
        assert_close(point2_harm_manual, point2_harm)
        assert_close(
            point2_manual, affine_invariant_mean_2points(point2_eucl, point2_harm)
        )

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_symmetry(self, n_features, cond, device, dtype, generator):
        """
        Test the symmetry property of the curve
        """
        points = random_SPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1_eucl = points[0]
        point1_harm = points[1]
        point2_eucl = points[2]
        point2_harm = points[3]
        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_12_auto, point_eucl_12_auto, point_harm_12_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )
        point_21_auto, point_eucl_21_auto, point_harm_21_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point2_eucl, point2_harm, point1_eucl, point1_harm, 1 - t
            )
        )
        point_12_manual, point_eucl_12_manual, point_harm_12_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )
        point_21_manual, point_eucl_21_manual, point_harm_21_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point2_eucl, point2_harm, point1_eucl, point1_harm, 1 - t
            )
        )

        assert_close(point_12_auto, point_21_auto)
        assert_close(point_eucl_12_auto, point_eucl_21_auto)
        assert_close(point_harm_12_auto, point_harm_21_auto)
        assert_close(point_12_manual, point_21_manual)
        assert_close(point_eucl_12_manual, point_eucl_21_manual)
        assert_close(point_harm_12_manual, point_harm_21_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identical_points(self, n_features, cond, device, dtype, generator):
        """
        Test that for t in [0,1], the curve between point and point returns point
        """
        # generate 1 SPD matrix
        point = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point_eucl = point[0]
        point_harm = point[1]
        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto, point_eucl_auto, point_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point_eucl, point_harm, point_eucl, point_harm, t
            )
        )
        point_manual, point_eucl_manual, point_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point_eucl, point_harm, point_eucl, point_harm, t
            )
        )

        assert_close(point_eucl_auto, point_eucl)
        assert_close(point_harm_auto, point_harm)
        assert_close(point_auto, affine_invariant_mean_2points(point_eucl, point_harm))

        assert_close(point_eucl_manual, point_eucl)
        assert_close(point_harm_manual, point_harm)
        assert_close(
            point_manual, affine_invariant_mean_2points(point_eucl, point_harm)
        )

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that curve between diagonal matrices works as expected
        """
        # random diagonal
        diag_points = random_DPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1_eucl = diag_points[0]
        diagvals1_eucl = point1_eucl.diagonal(dim1=-1, dim2=-2)
        point1_harm = diag_points[1]
        diagvals1_harm = point1_harm.diagonal(dim1=-1, dim2=-2)
        point2_eucl = diag_points[2]
        diagvals2_eucl = point2_eucl.diagonal(dim1=-1, dim2=-2)
        point2_harm = diag_points[2]
        diagvals2_harm = point2_harm.diagonal(dim1=-1, dim2=-2)
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto, point_eucl_auto, point_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )
        point_manual, point_eucl_manual, point_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )

        expected_diag_eucl = (1 - t) * diagvals1_eucl + t * diagvals2_eucl
        expected_diag_harm = 1 / ((1 - t) / diagvals1_harm + t / diagvals2_harm)
        expected_diag = torch.sqrt(expected_diag_eucl * expected_diag_harm)
        expected_eucl = torch.diag(expected_diag_eucl)
        expected_harm = torch.diag(expected_diag_harm)
        expected = torch.diag(expected_diag)

        assert_close(point_eucl_auto, expected_eucl)
        assert_close(point_harm_auto, expected_harm)
        assert_close(point_auto, expected)

        assert_close(point_eucl_manual, expected_eucl)
        assert_close(point_harm_manual, expected_harm)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that geodesic between 2 commuting matrices works as expected
        """
        # random diagonal
        diag_points = random_DPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diag_point1_eucl = diag_points[0]
        diagvals1_eucl = diag_point1_eucl.diagonal(dim1=-1, dim2=-2)
        diag_point1_harm = diag_points[1]
        diagvals1_harm = diag_point1_harm.diagonal(dim1=-1, dim2=-2)
        diag_point2_eucl = diag_points[2]
        diagvals2_eucl = diag_point2_eucl.diagonal(dim1=-1, dim2=-2)
        diag_point2_harm = diag_points[2]
        diagvals2_harm = diag_point2_harm.diagonal(dim1=-1, dim2=-2)
        # random eigvecs
        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # final points
        point1_eucl = eigvecs @ diag_point1_eucl @ eigvecs.transpose(-1, -2)
        point1_harm = eigvecs @ diag_point1_harm @ eigvecs.transpose(-1, -2)
        point2_eucl = eigvecs @ diag_point2_eucl @ eigvecs.transpose(-1, -2)
        point2_harm = eigvecs @ diag_point2_harm @ eigvecs.transpose(-1, -2)
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto, point_eucl_auto, point_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )
        point_manual, point_eucl_manual, point_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                point1_eucl, point1_harm, point2_eucl, point2_harm, t
            )
        )

        expected_diag_eucl = (1 - t) * diagvals1_eucl + t * diagvals2_eucl
        expected_diag_harm = 1 / ((1 - t) / diagvals1_harm + t / diagvals2_harm)
        expected_diag = torch.sqrt(expected_diag_eucl * expected_diag_harm)
        expected_eucl = (
            eigvecs @ torch.diag(expected_diag_eucl) @ eigvecs.transpose(-1, -2)
        )
        expected_harm = (
            eigvecs @ torch.diag(expected_diag_harm) @ eigvecs.transpose(-1, -2)
        )
        expected = eigvecs @ torch.diag(expected_diag) @ eigvecs.transpose(-1, -2)

        assert_close(point_eucl_auto, expected_eucl)
        assert_close(point_harm_auto, expected_harm)
        assert_close(point_auto, expected)

        assert_close(point_eucl_manual, expected_eucl)
        assert_close(point_harm_manual, expected_harm)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identity_to_general(self, n_features, cond, device, dtype, generator):
        """
        Test that the geodesic between the zero matrix and a random SPD works as expected
        """
        Id = torch.eye(n_features, device=device, dtype=dtype)
        # random point
        point = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point_eucl = point[0]
        point_harm = point[1]
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto, point_eucl_auto, point_harm_auto = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                Id, Id, point_eucl, point_harm, t
            )
        )
        point_manual, point_eucl_manual, point_harm_manual = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                Id, Id, point_eucl, point_harm, t
            )
        )

        expected_eucl = (1 - t) * Id + t * point_eucl
        expected_harm = torch.linalg.inv(
            (1 - t) * Id + t * torch.linalg.inv(point_harm)
        )
        expected = affine_invariant_mean_2points(expected_eucl, expected_harm)

        assert_close(point_eucl_auto, expected_eucl)
        assert_close(point_harm_auto, expected_harm)
        assert_close(point_auto, expected)

        assert_close(point_eucl_manual, expected_eucl)
        assert_close(point_harm_manual, expected_harm)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_features, cond, device, dtype, generator):
        """
        Test that backward works and that automatic and manual differentiation yield same results
        """
        # generate some random SPD matrices
        X = random_SPD(
            n_features,
            n_matrices=4,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        G_auto, _, _ = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_adaptive_update(
                X_auto[0], X_auto[1], X_auto[2], X_auto[3], t
            )
        )
        G_manual, _, _ = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicAdaptiveUpdate(
                X_manual[0], X_manual[1], X_manual[2], X_manual[3], t
            )
        )
        assert_close(G_auto, G_manual)

        loss_manual = torch.norm(G_manual)
        loss_manual.backward()
        loss_auto = torch.norm(G_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


class TestGeometricEuclideanHarmonicCurve:
    """
    Test suite for the geometric mean of euclidean geodescic and harmonic curve
    """

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_shape(self, n_features, cond, device, dtype, generator):
        """
        Test that geometric_euclidean_harmonic_curve function works and return appropriate shape
        """
        # generate 2 random SPD points
        points = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1 = points[0]
        point2 = points[1]
        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, t
        )
        point_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, t
        )

        assert point_auto.shape == point1.shape
        assert point_auto.device == point1.device
        assert point_auto.dtype == point1.dtype
        assert is_spd(point_auto)

        assert point_manual.shape == point1.shape
        assert point_manual.device == point1.device
        assert point_manual.dtype == point1.dtype
        assert is_spd(point_manual)

        assert_close(point_auto, point_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_extremities(self, n_features, cond, device, dtype, generator):
        """
        Test that for t=0 and t=1 we get point1 and point2
        """
        # generate 2 random SPD points
        points = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1 = points[0]
        point2 = points[1]

        point1_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, 0
        )
        point2_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, 1
        )
        point1_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, 0
        )
        point2_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, 1
        )

        assert_close(point1_auto, point1)
        assert_close(point2_auto, point2)
        assert_close(point1_manual, point1)
        assert_close(point2_manual, point2)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_symmetry(self, n_features, cond, device, dtype, generator):
        """
        Test the symmetry property of the curve
        """
        points = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1 = points[0]
        point2 = points[1]
        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_12_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, t
        )
        point_21_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point2, point1, 1 - t
        )
        point_12_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, t
        )
        point_21_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point2, point1, 1 - t
        )

        assert_close(point_12_auto, point_21_auto)
        assert_close(point_12_manual, point_21_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identical_points(self, n_features, cond, device, dtype, generator):
        """
        Test that for t in [0,1], the curve between point and point returns point
        """
        # generate 1 SPD matrix
        point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point, point, t
        )
        point_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point, point, t
        )

        assert_close(point_auto, point)
        assert_close(point_manual, point)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that geodesic between 2 diagonal matrices works as expected
        """
        # random diagonal
        diag_points = random_DPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        point1 = diag_points[0]
        diagvals1 = point1.diagonal(dim1=-1, dim2=-2)
        point2 = diag_points[1]
        diagvals2 = point2.diagonal(dim1=-1, dim2=-2)
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, t
        )
        point_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, t
        )

        expected_diag = torch.sqrt(
            ((1 - t) * diagvals1 + t * diagvals2)
            * 1
            / ((1 - t) * 1 / diagvals1 + t * 1 / diagvals2)
        )
        expected = torch.diag(expected_diag)

        assert_close(point_auto, expected)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that curve between 2 commuting matrices works as expected
        """
        # random diagonal
        diag_points = random_DPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diag_point1 = diag_points[0]
        diagvals1 = diag_point1.diagonal(dim1=-1, dim2=-2)
        diag_point2 = diag_points[1]
        diagvals2 = diag_point2.diagonal(dim1=-1, dim2=-2)
        # random eigvecs
        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # final points
        point1 = eigvecs @ diag_point1 @ eigvecs.transpose(-1, -2)
        point2 = eigvecs @ diag_point2 @ eigvecs.transpose(-1, -2)
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            point1, point2, t
        )
        point_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            point1, point2, t
        )

        expected_diag = torch.sqrt(
            ((1 - t) * diagvals1 + t * diagvals2)
            * 1
            / ((1 - t) * 1 / diagvals1 + t * 1 / diagvals2)
        )
        expected = eigvecs @ torch.diag(expected_diag) @ eigvecs.transpose(-1, -2)

        assert_close(point_auto, expected)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identity_to_general(self, n_features, cond, device, dtype, generator):
        """
        Test that the curve between the identity and a random SPD works as expected
        """
        Id = torch.eye(n_features, device=device, dtype=dtype)
        # random point
        point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        point_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            Id, point, t
        )
        point_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            Id, point, t
        )

        expected = affine_invariant_mean_2points(
            (1 - t) * Id + t * point,
            torch.linalg.inv((1 - t) * Id + t * torch.linalg.inv(point)),
        )

        assert_close(point_auto, expected)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_features, cond, device, dtype, generator):
        """
        Test that backward works and that automatic and manual differentiation yield same results
        """
        # generate some random SPD matrices
        X = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True
        # random t
        t = torch.rand(1, device=device, dtype=dtype, generator=generator)

        G_auto = kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve(
            X_auto[0], X_auto[1], t
        )
        G_manual = kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve(
            X_manual[0], X_manual[1], t
        )
        assert_close(G_auto, G_manual)

        loss_manual = torch.norm(G_manual)
        loss_manual.backward()
        loss_auto = torch.norm(G_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


class TestGeometricArithmeticHarmonicMean:
    """
    Test suite for the geometric mean of arithmetic and harmonic means
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that forward yields correct shape and structure (i.e., SPD solution)
        and that both harmonic_mean and HarmonicMean yield same result
        """
        # generate some SPD matrices
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # compute mean
        G_auto, G_auto_arith, G_auto_harm = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean(X)
        )
        G_manual, G_manual_arith, G_manual_harm = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean(X)
        )

        assert G_auto.shape == (n_features, n_features)
        assert G_auto.device == X.device
        assert G_auto.dtype == X.dtype
        assert is_spd(G_auto)
        assert G_auto_arith.shape == (n_features, n_features)
        assert G_auto_arith.device == X.device
        assert G_auto_arith.dtype == X.dtype
        assert is_spd(G_auto_arith)
        assert G_auto_harm.shape == (n_features, n_features)
        assert G_auto_harm.device == X.device
        assert G_auto_harm.dtype == X.dtype
        assert is_spd(G_auto_harm)

        assert G_manual.shape == (n_features, n_features)
        assert G_manual.device == X.device
        assert G_manual.dtype == X.dtype
        assert is_spd(G_manual)
        assert G_manual_arith.shape == (n_features, n_features)
        assert G_manual_arith.device == X.device
        assert G_manual_arith.dtype == X.dtype
        assert is_spd(G_manual_arith)
        assert G_manual_harm.shape == (n_features, n_features)
        assert G_manual_harm.device == X.device
        assert G_manual_harm.dtype == X.dtype
        assert is_spd(G_manual_harm)

        assert_close(G_auto_arith, G_manual_arith)
        assert_close(G_auto_harm, G_manual_harm)
        assert_close(G_auto, G_manual)

    @pytest.mark.parametrize("n_matrices, n_features", [(30, 100)])
    def test_identity_matrices(self, n_matrices, n_features, device, dtype):
        """
        Test that the mean of n_matrices identity matrices is the identity matrix
        """
        Id = torch.eye(n_features, device=device, dtype=dtype)
        data = Id.unsqueeze(0).repeat(n_matrices, 1, 1)

        G_auto, G_auto_arith, G_auto_harm = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean(data)
        )
        G_manual, G_manual_arith, G_manual_harm = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean(data)
        )

        assert_close(G_auto_arith, Id)
        assert_close(G_manual_arith, Id)
        assert_close(G_auto_harm, Id)
        assert_close(G_manual_harm, Id)
        assert_close(G_auto, Id)
        assert_close(G_manual, Id)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that the mean of diagonal matrices is working
        """
        diagmats = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals = diagmats.diagonal(dim1=-1, dim2=-2)

        G_auto, G_auto_arith, G_auto_harm = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean(diagmats)
        )
        G_manual, G_manual_arith, G_manual_harm = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean(diagmats)
        )

        expected_arith = torch.diag_embed(torch.mean(diagvals, dim=0))
        expected_harm = torch.diag_embed(1 / torch.mean(1 / diagvals, dim=0))
        expected = torch.diag_embed(
            torch.sqrt(torch.diag(expected_arith) * torch.diag(expected_harm))
        )

        assert_close(G_auto_arith, expected_arith)
        assert_close(G_manual_arith, expected_arith)
        assert_close(G_auto_harm, expected_harm)
        assert_close(G_manual_harm, expected_harm)
        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that the mean of commuting matrices works well
        """
        diagmats = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals = diagmats.diagonal(dim1=-1, dim2=-2)

        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        data = eigvecs @ diagmats @ eigvecs.transpose(-2, -1)

        G_auto, G_auto_arith, G_auto_harm = (
            kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean(data)
        )
        G_manual, G_manual_arith, G_manual_harm = (
            kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean(data)
        )

        expected_eigvals_arith = torch.diag_embed(torch.mean(diagvals, dim=0))
        expected_eigvals_harm = torch.diag_embed(1 / torch.mean(1 / diagvals, dim=0))
        expected_eigvals = torch.diag_embed(
            torch.sqrt(
                torch.diag(expected_eigvals_arith) * torch.diag(expected_eigvals_harm)
            )
        )
        expected_arith = eigvecs @ expected_eigvals_arith @ eigvecs.transpose(-2, -1)
        expected_harm = eigvecs @ expected_eigvals_harm @ eigvecs.transpose(-2, -1)
        expected = eigvecs @ expected_eigvals @ eigvecs.transpose(-2, -1)

        assert_close(G_auto_arith, expected_arith)
        assert_close(G_manual_arith, expected_arith)
        assert_close(G_auto_harm, expected_harm)
        assert_close(G_manual_harm, expected_harm)
        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that backward works and that automatic and manual differentiation yield same results
        """
        # generate some random SPD matrices
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        G_auto = kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean(
            X_auto
        )[0]
        G_manual = kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean(
            X_manual
        )[0]

        loss_manual = torch.norm(G_manual)
        loss_manual.backward()
        loss_auto = torch.norm(G_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

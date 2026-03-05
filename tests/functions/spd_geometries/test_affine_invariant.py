import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_geometries.affine_invariant as affine_invariant
import yetanotherspdnet.functions.spd_linalg as spd_linalg
from utils import is_spd, is_symmetric
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import arithmetic_mean
from yetanotherspdnet.random.spd import random_DPD, random_SPD
from yetanotherspdnet.random.stiefel import random_stiefel


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


class TestAffineInvariantGeodesic:
    """
    Test suite for affine-invariant geodesics
    """

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_shape(self, n_features, cond, device, dtype, generator):
        """
        Test that affine_invariant_geodesic function works and return appropriate shape
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

        point_auto = affine_invariant.affine_invariant_geodesic(point1, point2, t)
        point_manual = affine_invariant.AffineInvariantGeodesic.apply(point1, point2, t)

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

        point1_auto = affine_invariant.affine_invariant_geodesic(point1, point2, 0)
        point2_auto = affine_invariant.affine_invariant_geodesic(point1, point2, 1)

        point1_manual = affine_invariant.AffineInvariantGeodesic.apply(
            point1, point2, 0
        )
        point2_manual = affine_invariant.AffineInvariantGeodesic.apply(
            point1, point2, 1
        )

        assert_close(point1_auto, point1)
        assert_close(point2_auto, point2)
        assert_close(point1_manual, point1)
        assert_close(point2_manual, point2)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_symmetry(self, n_features, cond, device, dtype, generator):
        """
        Test the symmetry property of the geodesic
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

        point_12_auto = affine_invariant.affine_invariant_geodesic(point1, point2, t)
        point_21_auto = affine_invariant.affine_invariant_geodesic(
            point2, point1, 1 - t
        )
        point_12_manual = affine_invariant.AffineInvariantGeodesic.apply(
            point1, point2, t
        )
        point_21_manual = affine_invariant.AffineInvariantGeodesic.apply(
            point2, point1, 1 - t
        )

        assert_close(point_12_auto, point_21_auto)
        assert_close(point_12_manual, point_21_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identical_points(self, n_features, cond, device, dtype, generator):
        """
        Test that for t in [0,1], the geodesic between point and point returns point
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

        point_auto = affine_invariant.affine_invariant_geodesic(point, point, t)
        point_manual = affine_invariant.AffineInvariantGeodesic.apply(point, point, t)

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

        point_auto = affine_invariant.affine_invariant_geodesic(point1, point2, t)
        point_manual = affine_invariant.AffineInvariantGeodesic.apply(point1, point2, t)

        expected_diag = torch.pow(diagvals1, 1 - t) * torch.pow(diagvals2, t)
        expected = torch.diag(expected_diag)

        assert_close(point_auto, expected)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that geodesic between 2 commuting matrices works as expected
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

        point_auto = affine_invariant.affine_invariant_geodesic(point1, point2, t)
        point_manual = affine_invariant.AffineInvariantGeodesic.apply(point1, point2, t)

        expected_diag = torch.pow(diagvals1, 1 - t) * torch.pow(diagvals2, t)
        expected = eigvecs @ torch.diag(expected_diag) @ eigvecs.transpose(-1, -2)

        assert_close(point_auto, expected)
        assert_close(point_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identity_to_general(self, n_features, cond, device, dtype, generator):
        """
        Test that the geodesic between the identity and a random SPD works as expected
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

        point_auto = affine_invariant.affine_invariant_geodesic(Id, point, t)
        point_manual = affine_invariant.AffineInvariantGeodesic.apply(Id, point, t)

        expected = spd_linalg.powm_SPD(point, t)[0]

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

        G_auto = affine_invariant.affine_invariant_geodesic(X_auto[0], X_auto[1], t)
        G_manual = affine_invariant.AffineInvariantGeodesic.apply(
            X_manual[0], X_manual[1], t
        )

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


class TestAffineInvariantMean2points:
    """
    Test suite for the computation of the affine-invariant mean of 2 points,
    which is known in closed form
    """

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(self, n_features, cond, device, dtype, generator):
        """
        Test that forward yields correct shape and structure (SPD matrix)
        """
        # generate 2 SPD matrices
        X = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # compute mean
        G_auto = affine_invariant.affine_invariant_mean_2points(X[0], X[1])
        G_manual = affine_invariant.AffineInvariantMean2Points.apply(X[0], X[1])

        assert G_auto.shape == (n_features, n_features)
        assert G_auto.device == X.device
        assert G_auto.dtype == X.dtype
        assert is_spd(G_auto)

        assert G_manual.shape == (n_features, n_features)
        assert G_manual.device == X.device
        assert G_manual.dtype == X.dtype
        assert is_spd(G_manual)

        assert_close(G_auto, G_manual)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that the affine-invariant mean of diagonal matrices is the diagonal matrix
        whose elements are the geometric means of the diagonal matrices elements
        """
        diagmats = random_DPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals = diagmats.diagonal(dim1=-1, dim2=-2)

        G_auto = affine_invariant.affine_invariant_mean_2points(
            diagmats[0], diagmats[1]
        )
        G_manual = affine_invariant.AffineInvariantMean2Points.apply(
            diagmats[0], diagmats[1]
        )

        expected = torch.diag_embed(torch.pow(torch.prod(diagvals, dim=0), 1 / 2))

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(self, n_features, cond, device, dtype, generator):
        """
        Test that the affine-invariant mean of commuting matrices works well
        """
        diagmats = random_DPD(
            n_features,
            n_matrices=2,
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

        G_auto = affine_invariant.affine_invariant_mean_2points(data[0], data[1])
        G_manual = affine_invariant.AffineInvariantMean2Points.apply(data[0], data[1])

        expected_eigvals = torch.diag_embed(
            torch.pow(torch.prod(diagvals, dim=0), 1 / 2)
        )
        expected = eigvecs @ expected_eigvals @ eigvecs.transpose(-2, -1)

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_general_case(self, n_features, cond, device, dtype, generator):
        """
        Test affine-invariant mean in general case with exact solution
        """
        # generate a random mean
        G_true = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G_true_sqrtm, _, _ = spd_linalg.sqrtm_SPD(G_true)
        # generate random tangent vectors whose arithmetic mean is exactly zero
        tangent_vectors = spd_linalg.symmetrize(
            torch.randn(
                (2, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        tangent_vectors = tangent_vectors - arithmetic_mean(tangent_vectors)
        # multiply by some scale so that we don't get too far
        tangent_vectors = 0.1 * tangent_vectors
        # get SPD matrices from tangent vectors
        data = (
            G_true_sqrtm @ spd_linalg.expm_symmetric(tangent_vectors)[0] @ G_true_sqrtm
        )

        G_auto = affine_invariant.affine_invariant_mean_2points(data[0], data[1])
        G_manual = affine_invariant.AffineInvariantMean2Points.apply(data[0], data[1])

        assert_close(G_auto, G_true)
        assert_close(G_manual, G_true)

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

        G_manual = affine_invariant.AffineInvariantMean2Points.apply(
            X_manual[0], X_manual[1]
        )
        G_auto = affine_invariant.affine_invariant_mean_2points(X_auto[0], X_auto[1])

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

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_adequation_geodesic_fixedpoint(
        self, n_features, cond, device, dtype, generator
    ):
        """
        Test that we get the same result with affine_invariant_geodesic and affine-invariant mean
        """
        data = random_SPD(
            n_features,
            n_matrices=2,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        G_geodesic = affine_invariant.affine_invariant_geodesic(data[0], data[1], t=0.5)
        G_2points_auto = affine_invariant.affine_invariant_mean_2points(
            data[0], data[1]
        )
        G_fixedpoint_auto = affine_invariant.affine_invariant_mean(
            data, n_iterations=30
        )

        assert_close(G_geodesic, G_2points_auto)
        assert_close(G_fixedpoint_auto, G_2points_auto)

    # TODO: Also add comparison of gradients adequation between the 3


class TestAffineInvariantMean:
    """
    Test suite for the computation of the affine-invariant mean
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that forward yields correct shape and structure (i.e., SPD solution)
        and that both affine_invariant_mean and AffineInvariantMean yield same result
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
        G_auto = affine_invariant.affine_invariant_mean(X, n_iterations=20)
        G_manual = affine_invariant.AffineInvariantMean(X, n_iterations=20)

        assert G_auto.shape == (n_features, n_features)
        assert G_auto.device == X.device
        assert G_auto.dtype == X.dtype
        assert is_spd(G_auto)

        assert G_manual.shape == (n_features, n_features)
        assert G_manual.device == X.device
        assert G_manual.dtype == X.dtype
        assert is_spd(G_manual)

        assert_close(G_auto, G_manual)

    @pytest.mark.parametrize("n_matrices, n_features", [(30, 100)])
    # to check that it is stable with respect to iterations.
    # In this case, the algorithm is initialized with the solution
    @pytest.mark.parametrize("n_iterations", [1, 10])
    def test_identity_matrix(
        self, n_matrices, n_features, n_iterations, device, dtype, generator
    ):
        """
        Test that the mean of n_matrices whose mean is the identity matrix is the identity matrix
        """
        Id = torch.eye(n_features, device=device, dtype=dtype)
        # generate random tangent vectors whose arithmetic mean is exactly zero
        tangent_vectors = spd_linalg.symmetrize(
            torch.randn(
                (n_matrices, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        tangent_vectors = tangent_vectors - arithmetic_mean(tangent_vectors)
        # multiply by some scale so that we don't get too far
        tangent_vectors = 0.1 * tangent_vectors
        data = spd_linalg.expm_symmetric(tangent_vectors)[0]

        G_auto = affine_invariant.affine_invariant_mean(data, n_iterations)
        G_manual = affine_invariant.AffineInvariantMean(data, n_iterations)

        assert_close(G_auto, Id)
        assert_close(G_manual, Id)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    # to check that it is stable with respect to iterations.
    # In this case, only one iteration should be needed
    @pytest.mark.parametrize("n_iterations", [1, 10])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, n_iterations, device, dtype, generator
    ):
        """
        Test that the affine-invariant mean of diagonal matrices is the diagonal matrix
        whose elements are the geometric means of the diagonal matrices elements
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

        G_auto = affine_invariant.affine_invariant_mean(diagmats, n_iterations)
        G_manual = affine_invariant.AffineInvariantMean(diagmats, n_iterations)

        expected = torch.diag_embed(
            torch.pow(torch.prod(diagvals, dim=0), 1 / n_matrices)
        )

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    # to check that it is stable with respect to iterations.
    # In this case, only one iteration should be needed
    @pytest.mark.parametrize("n_iterations", [1, 10])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, n_iterations, device, dtype, generator
    ):
        """
        Test that the affine-invariant mean of commuting matrices works well
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

        G_auto = affine_invariant.affine_invariant_mean(data, n_iterations)
        G_manual = affine_invariant.AffineInvariantMean(data, n_iterations)

        expected_eigvals = torch.diag_embed(
            torch.pow(torch.prod(diagvals, dim=0), 1 / n_matrices)
        )
        expected = eigvecs @ expected_eigvals @ eigvecs.transpose(-2, -1)

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_general_case(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test affine-invariant mean in general case with exact solution
        """
        # generate a random mean
        G_true = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G_true_sqrtm, _, _ = spd_linalg.sqrtm_SPD(G_true)
        # generate random tangent vectors whose arithmetic mean is exactly zero
        tangent_vectors = spd_linalg.symmetrize(
            torch.randn(
                (n_matrices, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        tangent_vectors = tangent_vectors - arithmetic_mean(tangent_vectors)
        # multiply by some scale so that we don't get too far
        tangent_vectors = 0.1 * tangent_vectors
        # get SPD matrices from tangent vectors
        data = (
            G_true_sqrtm @ spd_linalg.expm_symmetric(tangent_vectors)[0] @ G_true_sqrtm
        )

        G_auto = affine_invariant.affine_invariant_mean(data, n_iterations=20)
        G_manual = affine_invariant.AffineInvariantMean(data, n_iterations=20)

        assert_close(G_auto, G_true)
        assert_close(G_manual, G_true)

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

        G_manual = affine_invariant.AffineInvariantMean(X_manual, n_iterations=10)
        G_auto = affine_invariant.affine_invariant_mean(X_auto, n_iterations=10)

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


class TestAffineInvariantStdScalar:
    """
    Test suite for the scalar standard deviation with respect to the affine-invariant distance
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_shape(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that affine_invariant_std_scalar and AffineInvariantStdScalar
        return correct shape, etc.
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
        G = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        std_auto = affine_invariant.affine_invariant_std_scalar(X, G)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(X, G)

        assert std_auto.shape == torch.Size([])
        assert std_auto.device == X.device
        assert std_auto.dtype == X.dtype
        assert std_auto >= 0

        assert std_manual.shape == torch.Size([])
        assert std_manual.device == X.device
        assert std_manual.dtype == X.dtype
        assert std_manual >= 0

        assert_close(std_auto, std_manual)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_zero_std(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that n_matrices same matrices have zero std when reference point is the considered matrix
        """
        G = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = torch.squeeze(G.clone().detach().repeat(n_matrices, 1, 1))

        std_auto = affine_invariant.affine_invariant_std_scalar(X, G)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(X, G)

        assert_close(std_auto, torch.tensor(0.0, device=device, dtype=dtype))
        assert_close(std_manual, torch.tensor(0.0, device=device, dtype=dtype))

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for diagonal matrices
        """
        X = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_X = X.diagonal(dim1=-1, dim2=-2)
        G = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_G = G.diagonal(dim1=-1, dim2=-2)

        std_auto = affine_invariant.affine_invariant_std_scalar(X, G)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(X, G)

        expected = torch.sqrt(
            torch.sum(torch.log(diagvals_X / diagvals_G) ** 2) / n_matrices
        )

        assert_close(std_auto, expected)
        assert_close(std_manual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for commuting matrices
        """
        diag_X = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_X = diag_X.diagonal(dim1=-1, dim2=-2)
        diag_G = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_G = diag_G.diagonal(dim1=-1, dim2=-2)

        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X = eigvecs @ diag_X @ eigvecs.transpose(-1, -2)
        G = eigvecs @ diag_G @ eigvecs.transpose(-1, -2)

        std_auto = affine_invariant.affine_invariant_std_scalar(X, G)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(X, G)

        expected = torch.sqrt(
            torch.sum(torch.log(diagvals_X / diagvals_G) ** 2) / n_matrices
        )

        assert_close(std_auto, expected)
        assert_close(std_manual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_general_case(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that general case works as expected
        """
        # generate a random mean
        G = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G_sqrtm = spd_linalg.sqrtm_SPD(G)[0]
        # generate random tangent vectors whose arithmetic mean is exactly zero
        tangent_vectors = spd_linalg.symmetrize(
            torch.randn(
                (n_matrices, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        tangent_vectors = tangent_vectors - arithmetic_mean(tangent_vectors)
        # multiply by some scale so that we don't get too far
        tangent_vectors = 0.1 * tangent_vectors
        # get SPD matrices from tangent vectors
        data = G_sqrtm @ spd_linalg.expm_symmetric(tangent_vectors)[0] @ G_sqrtm

        std_auto = affine_invariant.affine_invariant_std_scalar(data, G)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(data, G)

        expected = torch.sqrt(torch.linalg.norm(tangent_vectors) ** 2 / n_matrices)

        assert_close(std_auto, expected)
        assert_close(std_manual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that manual and automatic gradients are the same
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        G_manual = G.clone().detach()
        G_manual.requires_grad = True
        G_auto = G.clone().detach()
        G_auto.requires_grad = True

        std_auto = affine_invariant.affine_invariant_std_scalar(X_auto, G_auto)
        std_manual = affine_invariant.AffineInvariantStdScalar.apply(X_manual, G_manual)

        loss_manual = std_manual
        loss_manual.backward()
        loss_auto = std_auto
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert G_manual.grad is not None
        assert G_auto.grad is not None
        assert torch.isfinite(G_manual.grad).all()
        assert torch.isfinite(G_auto.grad).all()
        assert is_symmetric(G_manual.grad)
        assert is_symmetric(G_auto.grad)
        assert_close(G_manual.grad, G_auto.grad)


# ---------------------------------------------------
# Riemannian logarithm and exponential in coordinates
# ---------------------------------------------------
class TestAffineInvariantLogCoordinates:
    """
    Test suite for the Riemannian logarithm in coordinates
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test forward pass
        """
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X_auto = affine_invariant.affine_invariant_log_coordinates(
            data, reference_point
        )
        X_manual = affine_invariant.AffineInvariantLogCoordinates(data, reference_point)

        assert X_auto.dim() == data.dim() - 1
        if n_matrices > 1:
            assert X_auto.shape[0] == n_matrices
        assert X_auto.shape[-1] == n_features * (n_features + 1) // 2
        assert X_auto.device == data.device
        assert X_auto.dtype == data.dtype

        assert X_manual.dim() == data.dim() - 1
        if n_matrices > 1:
            assert X_manual.shape[0] == n_matrices
        assert X_manual.shape[-1] == n_features * (n_features + 1) // 2
        assert X_manual.device == data.device
        assert X_manual.dtype == data.dtype

        assert_close(X_auto, X_manual)
        assert_close(
            affine_invariant.affine_invariant_exp_coordinates(X_auto, reference_point),
            data,
        )
        assert_close(
            affine_invariant.affine_invariant_exp_coordinates(
                X_manual, reference_point
            ),
            data,
        )

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_identity_reference(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that Riemannian logarithm at identity is identical to LogmSPD
        """
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point = torch.eye(n_features, device=device, dtype=dtype)

        X_manual = affine_invariant.AffineInvariantLogCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_log_coordinates(
            data, reference_point
        )

        data_logm = spd_linalg.sym_matrix_to_coordinates(spd_linalg.LogmSPD.apply(data))

        assert_close(X_manual, data_logm)
        assert_close(X_auto, data_logm)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_self_log(self, n_features, cond, device, dtype, generator):
        """
        Test that the Riemannian logarithm of reference_point is zero
        """
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = affine_invariant.AffineInvariantLogCoordinates(
            reference_point, reference_point
        )
        X_auto = affine_invariant.affine_invariant_log_coordinates(
            reference_point, reference_point
        )

        assert_close(
            X_manual,
            torch.zeros(
                (n_features * (n_features + 1) // 2), device=device, dtype=dtype
            ),
        )
        assert_close(
            X_auto,
            torch.zeros(
                (n_features * (n_features + 1) // 2), device=device, dtype=dtype
            ),
        )

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for diagonal matrices
        """
        data = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_data = data.diagonal(dim1=-1, dim2=-2)
        reference_point = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_ref = reference_point.diagonal(dim1=-1, dim2=-2)

        X_manual = affine_invariant.AffineInvariantLogCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_log_coordinates(
            data, reference_point
        )

        expected = spd_linalg.sym_matrix_to_coordinates(
            torch.diag_embed(torch.log(diagvals_data / diagvals_ref))
        )

        assert_close(X_manual, expected)
        assert_close(X_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for commuting matrices
        """
        diagmats = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_data = diagmats.diagonal(dim1=-1, dim2=-2)

        diagref = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_ref = diagref.diagonal(dim1=-1, dim2=-2)

        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        data = eigvecs @ diagmats @ eigvecs.transpose(-2, -1)
        reference_point = eigvecs @ diagref @ eigvecs.transpose(-2, -1)

        X_manual = affine_invariant.AffineInvariantLogCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_log_coordinates(
            data, reference_point
        )

        expected = spd_linalg.sym_matrix_to_coordinates(
            eigvecs
            @ torch.diag_embed(torch.log(diagvals_data / diagvals_ref))
            @ eigvecs.transpose(-2, -1)
        )

        assert_close(X_manual, expected)
        assert_close(X_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_general_case(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that it works as expected in general case
        """
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point_sqrtm = spd_linalg.sqrtm_SPD(reference_point)[0]
        tangent_vectors = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        data = (
            reference_point_sqrtm
            @ spd_linalg.expm_symmetric(tangent_vectors)[0]
            @ reference_point_sqrtm
        )

        X_manual = affine_invariant.AffineInvariantLogCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_log_coordinates(
            data, reference_point
        )

        expected = spd_linalg.sym_matrix_to_coordinates(tangent_vectors)

        assert_close(X_manual, expected, rtol=1e-5, atol=1e-4)
        assert_close(X_auto, expected, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that backward works as expected
        """
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = data.clone().detach()
        X_manual.requires_grad = True
        X_auto = data.clone().detach()
        X_auto.requires_grad = True

        G_manual = reference_point.clone().detach()
        G_manual.requires_grad = True
        G_auto = reference_point.clone().detach()
        G_auto.requires_grad = True

        Y_manual = affine_invariant.AffineInvariantLogCoordinates(X_manual, G_manual)
        Y_auto = affine_invariant.affine_invariant_log_coordinates(X_auto, G_auto)

        loss_manual = torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(Y_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert G_manual.grad is not None
        assert G_auto.grad is not None
        assert torch.isfinite(G_manual.grad).all()
        assert torch.isfinite(G_auto.grad).all()
        assert is_symmetric(G_manual.grad)
        assert is_symmetric(G_auto.grad)
        assert_close(G_manual.grad, G_auto.grad)


class TestAffineInvariantExpCoordinates:
    """
    Test suite for the Riemannian exponential from coordinates
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test forward pass
        """
        data = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            data, reference_point
        )
        X_manual = affine_invariant.AffineInvariantExpCoordinates(data, reference_point)

        assert X_auto.dim() == data.dim() + 1
        if n_matrices > 1:
            assert X_auto.shape[0] == n_matrices
        assert X_auto.shape[-1] == n_features
        assert X_auto.shape[-2] == n_features
        assert X_auto.device == data.device
        assert X_auto.dtype == data.dtype

        assert X_manual.dim() == data.dim() + 1
        if n_matrices > 1:
            assert X_manual.shape[0] == n_matrices
        assert X_manual.shape[-1] == n_features
        assert X_manual.shape[-2] == n_features
        assert X_manual.device == data.device
        assert X_manual.dtype == data.dtype

        assert_close(X_auto, X_manual)
        assert_close(
            affine_invariant.affine_invariant_log_coordinates(X_auto, reference_point),
            data,
            atol=1e-4,
            rtol=1e-5,
        )
        assert_close(
            affine_invariant.affine_invariant_log_coordinates(
                X_manual, reference_point
            ),
            data,
            atol=1e-4,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features", [100])
    def test_identity_reference(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that Riemannian exponential at identity is identical to ExpmSymmetric
        """
        data = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        reference_point = torch.eye(n_features, device=device, dtype=dtype)

        X_manual = affine_invariant.AffineInvariantExpCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            data, reference_point
        )

        data_expm = spd_linalg.ExpmSymmetric.apply(
            spd_linalg.sym_coordinates_to_matrix(data, n_features)
        )

        assert_close(X_manual, data_expm)
        assert_close(X_auto, data_expm)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_self_exp(self, n_features, cond, device, dtype, generator):
        """
        Test that the Riemannian exponential of zero is reference_point
        """
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        zero_vec = torch.zeros(
            (n_features * (n_features + 1) // 2), device=device, dtype=dtype
        )
        X_manual = affine_invariant.AffineInvariantExpCoordinates(
            zero_vec, reference_point
        )
        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            zero_vec, reference_point
        )

        assert_close(X_manual, reference_point)
        assert_close(X_auto, reference_point)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for diagonal matrices
        """
        diagvals_data = torch.squeeze(
            torch.randn((n_matrices, n_features), device=device, dtype=dtype)
        )
        data = spd_linalg.sym_matrix_to_coordinates(torch.diag_embed(diagvals_data))
        reference_point = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_ref = reference_point.diagonal(dim1=-1, dim2=-2)

        X_manual = affine_invariant.AffineInvariantExpCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            data, reference_point
        )

        expected = torch.diag_embed(torch.exp(diagvals_data) * diagvals_ref)

        assert_close(X_manual, expected)
        assert_close(X_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for commuting matrices
        """
        diagvals_data = torch.squeeze(
            torch.randn((n_matrices, n_features), device=device, dtype=dtype)
        )
        diagmats = torch.diag_embed(diagvals_data)

        diagref = random_DPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diagvals_ref = diagref.diagonal(dim1=-1, dim2=-2)

        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        data = spd_linalg.sym_matrix_to_coordinates(
            eigvecs @ diagmats @ eigvecs.transpose(-2, -1)
        )
        reference_point = eigvecs @ diagref @ eigvecs.transpose(-2, -1)

        X_manual = affine_invariant.AffineInvariantExpCoordinates(data, reference_point)
        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            data, reference_point
        )

        expected = (
            eigvecs
            @ torch.diag_embed(torch.exp(diagvals_data) * diagvals_ref)
            @ eigvecs.transpose(-2, -1)
        )

        assert_close(X_manual, expected)
        assert_close(X_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_general_case(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that it works as expected in general case
        """
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point_inv_sqrtm = spd_linalg.inv_sqrtm_SPD(reference_point)[0]
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        tangent_vectors = spd_linalg.sym_matrix_to_coordinates(
            spd_linalg.logm_SPD(
                reference_point_inv_sqrtm @ data @ reference_point_inv_sqrtm
            )[0]
        )

        X_manual = affine_invariant.AffineInvariantExpCoordinates(
            tangent_vectors, reference_point
        )
        X_auto = affine_invariant.affine_invariant_exp_coordinates(
            tangent_vectors, reference_point
        )

        assert_close(X_manual, data)
        assert_close(X_auto, data)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that backward works as expected
        """
        tangent_vectors = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        reference_point = random_SPD(
            n_features,
            n_matrices=1,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = tangent_vectors.clone().detach()
        X_manual.requires_grad = True
        X_auto = tangent_vectors.clone().detach()
        X_auto.requires_grad = True

        G_manual = reference_point.clone().detach()
        G_manual.requires_grad = True
        G_auto = reference_point.clone().detach()
        G_auto.requires_grad = True

        Y_manual = affine_invariant.AffineInvariantExpCoordinates(X_manual, G_manual)
        Y_auto = affine_invariant.affine_invariant_exp_coordinates(X_auto, G_auto)

        loss_manual = torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(Y_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

        assert G_manual.grad is not None
        assert G_auto.grad is not None
        assert torch.isfinite(G_manual.grad).all()
        assert torch.isfinite(G_auto.grad).all()
        assert is_symmetric(G_manual.grad)
        assert is_symmetric(G_auto.grad)
        assert_close(G_manual.grad, G_auto.grad)


class TestAffineInvariantParallelTransportCoordinates:
    """
    Test suite for parallel transport according to affine-invariant geometry
    """

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward_shape(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that output structure is as expected
        """
        # generate reference SPD point and new reference SPD point
        reference_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        new_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )

        # transported coordinates
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, new_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, new_point
            )
        )

        assert coordinates_new_manual.shape == coordinates_reference.shape
        assert coordinates_new_manual.device == coordinates_reference.device
        assert coordinates_new_manual.dtype == coordinates_reference.dtype

        assert coordinates_new_auto.shape == coordinates_reference.shape
        assert coordinates_new_auto.device == coordinates_reference.device
        assert coordinates_new_auto.dtype == coordinates_reference.dtype

        assert_close(coordinates_new_auto, coordinates_new_manual)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_same_reference_points(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that not changing reference point leaves tangent vector unchanged
        """
        # generate reference SPD point
        reference_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transport
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, reference_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, reference_point
            )
        )
        # check that there is no changes
        assert_close(coordinates_new_manual, coordinates_reference)
        assert_close(coordinates_reference, coordinates_new_auto)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_from_identity(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected from identity matrices
        """
        # identity matrices
        Identity = torch.diag_embed(
            torch.squeeze(
                torch.ones((n_matrices, n_features), device=device, dtype=dtype)
            )
        )
        # new reference point
        new_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at identity
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transport
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, Identity, new_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, Identity, new_point
            )
        )
        # expected
        # In coordinates, expected is the same as coordinates_reference
        expected = coordinates_reference

        assert_close(coordinates_new_manual, expected)
        assert_close(coordinates_new_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_to_identity(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that it works as expected to identity matrices
        """
        # generate reference SPD point
        reference_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # identity matrices
        Identity = torch.diag_embed(
            torch.squeeze(
                torch.ones((n_matrices, n_features), device=device, dtype=dtype)
            )
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transport
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, Identity
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, Identity
            )
        )
        # expected
        # In coordinates, expected is the same as coordinates_reference
        expected = coordinates_reference

        assert_close(coordinates_new_manual, expected)
        assert_close(coordinates_new_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_diagonal_reference_points(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for diagonal reference points
        """
        reference_point = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        new_point = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transport
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, new_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, new_point
            )
        )
        # expected
        # In coordinates, expected is the same as coordinates_reference
        expected = coordinates_reference

        assert_close(coordinates_new_manual, expected)
        assert_close(coordinates_new_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_commuting_reference_points(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that it works as expected for commuting reference points
        """
        diag_reference = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        diag_new = random_DPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        eigvecs = random_stiefel(
            n_features,
            n_features,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_point = eigvecs @ diag_reference @ eigvecs.transpose(-2, -1)
        new_point = eigvecs @ diag_new @ eigvecs.transpose(-2, -1)
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transport
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, new_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, new_point
            )
        )
        # expected
        # In coordinates, expected is the same as coordinates_reference
        expected = coordinates_reference

        assert_close(coordinates_new_manual, expected)
        assert_close(coordinates_new_auto, expected)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_invariance_properties(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that the norm is preserved and that transporting a vector back to
        original tangent space yield original vector
        """
        # generate reference SPD point and new reference SPD point
        reference_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        new_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # transported coordinates
        coordinates_new_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_reference, reference_point, new_point
            )
        )
        coordinates_new_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_reference, reference_point, new_point
            )
        )
        # transport coordinates back to original reference_point
        coordinates_original_manual = (
            affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
                coordinates_new_manual, new_point, reference_point
            )
        )
        coordinates_original_auto = (
            affine_invariant.affine_invariant_parallel_transport_coordinates(
                coordinates_new_auto, new_point, reference_point
            )
        )
        # check that it changed
        assert not torch.allclose(coordinates_new_manual, coordinates_reference)
        assert not torch.allclose(coordinates_new_auto, coordinates_reference)
        # check norm invariance
        assert_close(
            torch.norm(coordinates_new_manual), torch.norm(coordinates_reference)
        )
        assert_close(
            torch.norm(coordinates_new_auto), torch.norm(coordinates_reference)
        )
        # check going back yields original coordinates
        assert_close(coordinates_original_manual, coordinates_reference)
        assert_close(coordinates_original_auto, coordinates_reference)

    @pytest.mark.parametrize("n_matrices", [1, 30])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that manual and automatic differentiation yield same results
        """
        # generate reference SPD point and new reference SPD point
        reference_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        new_point = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # generate random coordinates at reference_point
        coordinates_reference = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        X_manual = reference_point.clone().detach()
        X_manual.requires_grad = True
        X_auto = reference_point.clone().detach()
        X_auto.requires_grad = True

        Y_manual = new_point.clone().detach()
        Y_manual.requires_grad = True
        Y_auto = new_point.clone().detach()
        Y_auto.requires_grad = True

        xi_manual = coordinates_reference.clone().detach()
        xi_manual.requires_grad = True
        xi_auto = coordinates_reference.clone().detach()
        xi_auto.requires_grad = True

        eta_manual = affine_invariant.AffineInvariantParallelTransportCoordinates.apply(
            xi_manual, X_manual, Y_manual
        )
        eta_auto = affine_invariant.affine_invariant_parallel_transport_coordinates(
            xi_auto, X_auto, Y_auto
        )

        loss_manual = torch.norm(eta_manual)
        loss_manual.backward()
        loss_auto = torch.norm(eta_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert Y_manual.grad is not None
        assert Y_auto.grad is not None
        assert torch.isfinite(Y_manual.grad).all()
        assert torch.isfinite(Y_auto.grad).all()
        assert is_symmetric(Y_manual.grad)
        assert is_symmetric(Y_auto.grad)
        assert_close(Y_manual.grad, Y_auto.grad)

        assert xi_manual.grad is not None
        assert xi_auto.grad is not None
        assert torch.isfinite(xi_manual.grad).all()
        assert torch.isfinite(xi_auto.grad).all()
        assert_close(xi_manual.grad, xi_auto.grad)

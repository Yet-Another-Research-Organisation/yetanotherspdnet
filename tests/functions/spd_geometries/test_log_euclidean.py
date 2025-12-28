import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_geometries.log_euclidean as log_euclidean
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


class TestLogEuclideanGeodesic:
    """
    Test suite for log-Euclidean geodesics
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

        point_auto = log_euclidean.log_euclidean_geodesic(point1, point2, t)
        point_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, t)

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

        point1_auto = log_euclidean.log_euclidean_geodesic(point1, point2, 0)
        point2_auto = log_euclidean.log_euclidean_geodesic(point1, point2, 1)
        point1_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, 0)
        point2_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, 1)

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

        point_12_auto = log_euclidean.log_euclidean_geodesic(point1, point2, t)
        point_21_auto = log_euclidean.log_euclidean_geodesic(point2, point1, 1 - t)
        point_12_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, t)
        point_21_manual = log_euclidean.LogEuclideanGeodesic(point2, point1, 1 - t)

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

        point_auto = log_euclidean.log_euclidean_geodesic(point, point, t)
        point_manual = log_euclidean.LogEuclideanGeodesic(point, point, t)

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

        point_auto = log_euclidean.log_euclidean_geodesic(point1, point2, t)
        point_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, t)

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

        point_auto = log_euclidean.log_euclidean_geodesic(point1, point2, t)
        point_manual = log_euclidean.LogEuclideanGeodesic(point1, point2, t)

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

        point_auto = log_euclidean.log_euclidean_geodesic(Id, point, t)
        point_manual = log_euclidean.LogEuclideanGeodesic(Id, point, t)

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

        G_auto = log_euclidean.log_euclidean_geodesic(X_auto[0], X_auto[1], t)
        G_manual = log_euclidean.LogEuclideanGeodesic(X_manual[0], X_manual[1], t)
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


# ------------------
# Log-Euclidean mean
# ------------------
class TestLogEuclideanMean:
    """
    Test suite for log-Euclidean mean
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
        G_auto = log_euclidean.log_euclidean_mean(X)
        G_manual = log_euclidean.LogEuclideanMean(X)

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
    def test_identity_matrix(self, n_matrices, n_features, device, dtype, generator):
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

        G_auto = log_euclidean.log_euclidean_mean(data)
        G_manual = log_euclidean.LogEuclideanMean(data)

        assert_close(G_auto, Id)
        assert_close(G_manual, Id)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_diagonal_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that the log-Euclidean mean of diagonal matrices is the diagonal matrix
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

        G_auto = log_euclidean.log_euclidean_mean(diagmats)
        G_manual = log_euclidean.LogEuclideanMean(diagmats)

        expected = torch.diag_embed(
            torch.pow(torch.prod(diagvals, dim=0), 1 / n_matrices)
        )

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_commuting_matrices(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that the log-Euclidean mean of commuting matrices works well
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

        G_auto = log_euclidean.log_euclidean_mean(data)
        G_manual = log_euclidean.LogEuclideanMean(data)

        expected_eigvals = torch.diag_embed(
            torch.pow(torch.prod(diagvals, dim=0), 1 / n_matrices)
        )
        expected = eigvecs @ expected_eigvals @ eigvecs.transpose(-2, -1)

        assert_close(G_auto, expected)
        assert_close(G_manual, expected)

    @pytest.mark.parametrize("n_matrices, n_features, cond", [(30, 100, 1000)])
    def test_general_case(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test log-Euclidean mean in general case with exact solution
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
        data = spd_linalg.expm_symmetric(
            tangent_vectors + spd_linalg.logm_SPD(G_true)[0]
        )[0]

        G_auto = log_euclidean.log_euclidean_mean(data)
        G_manual = log_euclidean.LogEuclideanMean(data)

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

        G_auto = log_euclidean.log_euclidean_mean(X_auto)
        G_manual = log_euclidean.LogEuclideanMean(X_manual)

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


class TestLogEuclideanStdScalar:
    """
    Test suite for the scalar standard deviation with respect to the log-Euclidean distance
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

        std_auto = log_euclidean.log_euclidean_std_scalar(X, G)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(X, G)

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

        std_auto = log_euclidean.log_euclidean_std_scalar(X, G)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(X, G)

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

        std_auto = log_euclidean.log_euclidean_std_scalar(X, G)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(X, G)

        expected = torch.sqrt(
            torch.sum((torch.log(diagvals_X) - torch.log(diagvals_G)) ** 2) / n_matrices
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

        std_auto = log_euclidean.log_euclidean_std_scalar(X, G)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(X, G)

        expected = torch.sqrt(
            torch.sum((torch.log(diagvals_X) - torch.log(diagvals_G)) ** 2) / n_matrices
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
        G_logm = spd_linalg.logm_SPD(G)[0]
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
        data = spd_linalg.expm_symmetric(tangent_vectors + G_logm)[0]

        std_auto = log_euclidean.log_euclidean_std_scalar(data, G)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(data, G)

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

        std_auto = log_euclidean.log_euclidean_std_scalar(X_auto, G_auto)
        std_manual = log_euclidean.LogEuclideanStdScalar.apply(X_manual, G_manual)

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

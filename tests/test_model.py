import pytest
import torch
from torch.nn.utils import parametrize
from torch.testing import assert_close

# from utils import is_symmetric
from yetanotherspdnet.model import SPDnet
from yetanotherspdnet.nn.base import BiMap, LogEig, ReEig, Vec, Vech
from yetanotherspdnet.nn.batchnorm import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
)
from yetanotherspdnet.random.spd import random_SPD


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float32


@pytest.fixture(scope="function")
def generator(device):
    gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    gen.manual_seed(777)
    return gen


class TestSPDnet:
    """Test suite for SPDnet module"""

    @pytest.mark.parametrize(
        "input_dim, hidden_layers, output_dim",
        [
            (10, [5], 3),
            (20, [15, 10], 5),
            (30, [20, 15, 10], 10),
        ],
    )
    def test_initialization(
        self, input_dim, hidden_layers, output_dim, device, dtype, generator
    ):
        """Test that SPDNet initializes correctly"""
        model = SPDnet(
            input_dim=input_dim,
            hidden_layers_size=hidden_layers,
            output_dim=output_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert model.input_dim == input_dim
        assert model.hidden_layers_size == hidden_layers
        assert model.output_dim == output_dim
        assert model.device.type == device.type
        assert model.dtype == dtype

    @pytest.mark.parametrize(
        "input_dim, hidden_layers, output_dim",
        [
            (10, [5], 3),
            (20, [15, 10], 5),
        ],
    )
    @pytest.mark.parametrize("n_samples", [1, 10])
    def test_forward_shape(
        self, input_dim, hidden_layers, output_dim, n_samples, device, dtype, generator
    ):
        """Test that forward pass returns correct shape"""
        model = SPDnet(
            input_dim=input_dim,
            hidden_layers_size=hidden_layers,
            output_dim=output_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X = random_SPD(
            input_dim, n_samples, device=device, dtype=dtype, generator=generator
        )
        output = model(X)

        expected_shape = (n_samples, output_dim) if n_samples > 1 else (output_dim,)
        assert output.shape == expected_shape
        assert output.dtype == dtype
        assert output.device.type == device.type

    @pytest.mark.parametrize("vec_type", ["vec", "vech"])
    def test_vec_type(self, vec_type, device, dtype, generator):
        """Test different vectorization types"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[5],
            output_dim=3,
            vec_type=vec_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert model.vec_type == vec_type
        if vec_type == "vec":
            assert isinstance(model.vectorization, Vec)
            assert model.linear.in_features == 5**2
        elif vec_type == "vech":
            assert isinstance(model.vectorization, Vech)
            assert model.linear.in_features == 5 * 6 // 2

        # Test forward pass works
        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        output = model(X)
        assert output.shape == (5, 3)

    @pytest.mark.parametrize("softmax", [True, False])
    def test_softmax(self, softmax, device, dtype, generator):
        """Test softmax activation"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[5],
            output_dim=3,
            softmax=softmax,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        output = model(X)

        if softmax:
            # Check that outputs sum to 1 along last dimension
            assert_close(output.sum(dim=-1), torch.ones(5, device=device, dtype=dtype))
            # Check that all outputs are in [0, 1]
            assert (output >= 0).all() and (output <= 1).all()
        else:
            # Without softmax, outputs are not constrained
            assert hasattr(model, "softmax_layer") is False or model.softmax is False

    @pytest.mark.parametrize("batchnorm", [True, False])
    def test_batchnorm(self, batchnorm, device, dtype, generator):
        """Test batch normalization"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            batchnorm=batchnorm,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Count BatchNormSPDMean layers
        batchnorm_count = sum(
            1 for layer in model.spdnet_layers if isinstance(layer, BatchNormSPDMean)
        )

        if batchnorm:
            # Should have one BatchNorm per hidden layer
            assert batchnorm_count == len(model.hidden_layers_size)
        else:
            assert batchnorm_count == 0

        # Test forward pass works
        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        output = model(X)
        assert output.shape == (5, 3)

    def test_layer_count(self, device, dtype, generator):
        """Test correct number of layers"""
        hidden_layers = [8, 6, 4]
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=hidden_layers,
            output_dim=3,
            batchnorm=False,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Expected: BiMap + ReEig for each hidden layer + LogEig at end
        expected_layers = len(hidden_layers) * 2 + 1
        assert len(model.spdnet_layers) == expected_layers

        # Count layer types using isinstance (more robust)
        bimap_count = sum(
            1 for layer in model.spdnet_layers if isinstance(layer, BiMap)
        )
        reeig_count = sum(
            1 for layer in model.spdnet_layers if isinstance(layer, ReEig)
        )
        logeig_count = sum(
            1 for layer in model.spdnet_layers if isinstance(layer, LogEig)
        )

        assert bimap_count == len(hidden_layers)
        assert reeig_count == len(hidden_layers)
        assert logeig_count == 1

    @pytest.mark.parametrize("batchnorm_type", ["mean_only", "mean_var_scalar"])
    def test_layer_count_with_batchnorm(self, batchnorm_type, device, dtype, generator):
        """Test layer count with batch normalization"""
        hidden_layers = [8, 6]
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=hidden_layers,
            output_dim=3,
            batchnorm=True,
            batchnorm_type=batchnorm_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Expected: (BiMap + ReEig + BatchNorm) for each hidden layer + LogEig
        # = 2 * 3 + 1 = 7 layers
        expected_layers = len(hidden_layers) * 3 + 1
        assert len(model.spdnet_layers) == expected_layers

    @pytest.mark.parametrize("vec_type", ["vec", "vech"])
    @pytest.mark.parametrize("batchnorm_type", ["mean_only", "mean_var_scalar"])
    def test_use_autograd(self, vec_type, batchnorm_type, device, dtype, generator):
        """Test that autograd and manual gradient give same results"""
        # Create two models with same initialization
        gen1 = torch.Generator(device=device)
        gen1.manual_seed(777)
        model_manual = SPDnet(
            input_dim=10,
            hidden_layers_size=[5],
            output_dim=3,
            batchnorm=True,
            batchnorm_type=batchnorm_type,
            vec_type=vec_type,
            use_autograd=False,
            device=device,
            dtype=dtype,
            generator=gen1,
        )

        gen2 = torch.Generator(device=device)
        gen2.manual_seed(777)
        model_autograd = SPDnet(
            input_dim=10,
            hidden_layers_size=[5],
            output_dim=3,
            batchnorm=True,
            batchnorm_type=batchnorm_type,
            vec_type=vec_type,
            use_autograd=True,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        # Copy weights to ensure both models have identical parameters
        for (_name1, param1), (_name2, param2) in zip(
            model_manual.named_parameters(), model_autograd.named_parameters()
        ):
            param2.data.copy_(param1.data)

        # Create input
        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_autograd = X.clone().detach()
        X_autograd.requires_grad = True

        # Forward pass
        output_manual = model_manual(X_manual)
        output_autograd = model_autograd(X_autograd)

        # Check forward outputs are the same
        assert_close(output_manual, output_autograd)

        # Backward pass
        loss_manual = output_manual.sum()
        loss_manual.backward()

        loss_autograd = output_autograd.sum()
        loss_autograd.backward()

        # Check input gradients
        assert X_manual.grad is not None
        assert X_autograd.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_autograd.grad).all()
        assert_close(X_manual.grad, X_manual.grad.transpose(-1, -2))
        assert_close(X_autograd.grad, X_autograd.grad.transpose(-1, -2))
        # assert is_symmetric(X_manual.grad)
        # assert is_symmetric(X_autograd.grad)

        # Compare gradients - should be the same
        assert_close(X_manual.grad, X_autograd.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("vec_type", ["vec", "vech"])
    @pytest.mark.parametrize("batchnorm_type", ["mean_only", "mean_var_scalar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, vec_type, batchnorm_type, use_autograd, device, dtype, generator
    ):
        """Test that gradients flow through the network"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            batchnorm=True,
            batchnorm_type=batchnorm_type,
            vec_type=vec_type,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )

        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        X.requires_grad = True

        output = model(X)
        loss = output.sum()
        loss.backward()

        # Check input gradient
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()
        assert_close(X.grad, X.grad.transpose(-1, -2))
        # assert is_symmetric(X.grad)

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_parameter_count(self, device, dtype, generator):
        """Test that model has expected number of parameters"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have: BiMap weights (10x8 + 8x6) + Linear (6*6 x 3 + 3)
        expected = (10 * 8 + 8 * 6) + (36 * 3 + 3)
        assert n_params == expected

    def test_get_last_tensor(self, device, dtype, generator):
        """Test get_last_tensor method"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)

        # Get last tensor (before vectorization and linear layer)
        last_tensor = model.get_last_tensor(X)

        # Should be symmetric matrices of size 6x6
        assert last_tensor.shape == (5, 6, 6)
        # Check symmetry with relaxed tolerance for numerical stability across PyTorch versions
        assert torch.allclose(last_tensor, last_tensor.transpose(-1, -2), atol=1e-5, rtol=1e-5)

    def test_train_eval_mode(self, device, dtype, generator):
        """Test train/eval mode switching"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            batchnorm=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

        # Test that forward pass works in both modes
        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)

        model.train()
        output_train = model(X)

        model.eval()
        output_eval = model(X)

        # Outputs should be different with batchnorm
        assert not torch.allclose(output_train, output_eval)

    def test_repr_and_str(self, device, dtype, generator):
        """Test string representations"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        repr_str = repr(model)
        str_str = str(model)

        assert "SPDnet" in repr_str
        assert "input_dim=10" in repr_str
        assert "hidden_layers_size=[8, 6]" in repr_str
        assert "output_dim=3" in repr_str
        assert repr_str == str_str

    def test_layers_str(self, device, dtype, generator):
        """Test layers_str method"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layers_str = model.layers_str()

        assert "SPDnet Layers:" in layers_str
        assert "BiMap" in layers_str
        assert "ReEig" in layers_str
        assert "LogEig" in layers_str
        assert "Linear" in layers_str

    def test_model_hash(self, device, dtype, generator):
        """Test model hash creation and retrieval"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Create hash
        hash1 = model.create_model_name_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 8  # CRC32 hash is 8 hex characters

        # Get hash (should return same)
        hash2 = model.get_model_hash()
        assert hash1 == hash2

        # Different models should have different hashes
        model2 = SPDnet(
            input_dim=10,
            hidden_layers_size=[5],  # Different
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        hash3 = model2.create_model_name_hash()
        assert hash1 != hash3

    def test_reproducibility_with_generator(self, device, dtype):
        """Test that same generator gives same initialization"""
        gen1 = torch.Generator(device=device).manual_seed(42)
        model1 = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=gen1,
        )

        gen2 = torch.Generator(device=device).manual_seed(42)
        model2 = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        # Check that BiMap weights are the same
        for (name1, param1), (_name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if "weight" in name1 and "spdnet_layers" in name1:
                assert_close(param1, param2)

    @pytest.mark.parametrize("bimap_parametrized", [True, False])
    def test_bimap_parametrization(self, bimap_parametrized, device, dtype, generator):
        """Test BiMap parametrization option"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            bimap_parametrized=bimap_parametrized,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Check first BiMap layer
        first_bimap = model.spdnet_layers[0]
        assert isinstance(first_bimap, BiMap)

        if bimap_parametrized:
            assert parametrize.is_parametrized(first_bimap, "weight")
        else:
            assert not parametrize.is_parametrized(first_bimap, "weight")

    def test_reeig_eps(self, device, dtype, generator):
        """Test ReEig epsilon parameter"""
        eps = 1e-2
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            reeig_eps=eps,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert model.reeig_eps == eps

        # Check that ReEig layers have correct eps
        reeig_layers = [
            layer for layer in model.spdnet_layers if isinstance(layer, ReEig)
        ]
        for layer in reeig_layers:
            assert layer.eps == eps

    def test_single_sample_input(self, device, dtype, generator):
        """Test with single sample (2D input)"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Single SPD matrix (2D)
        X = random_SPD(10, 1, device=device, dtype=dtype, generator=generator).squeeze(
            0
        )
        output = model(X)

        assert output.shape == (3,)

    def test_batch_input(self, device, dtype, generator):
        """Test with batch input"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8],
            output_dim=3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        batch_size = 32
        X = random_SPD(10, batch_size, device=device, dtype=dtype, generator=generator)
        output = model(X)

        assert output.shape == (batch_size, 3)

    @pytest.mark.parametrize("batchnorm_type", ["mean_only", "mean_var_scalar"])
    def test_gradient_flow_all_layers(self, batchnorm_type, device, dtype, generator):
        """Test that gradients flow through all layers"""
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            batchnorm=True,
            batchnorm_type=batchnorm_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        X = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        X.requires_grad = True

        output = model(X)
        loss = output.mean()
        loss.backward()

        # Check all BiMap weights and BatchNorm Covbias (and stdScalarbias) have gradients
        for i, layer in enumerate(model.spdnet_layers):
            if isinstance(layer, BiMap):
                assert layer.parametrizations.weight.original.grad is not None, (
                    f"BiMap layer {i} has no gradient"
                )
                assert layer.parametrizations.weight.original.grad.abs().sum() > 0, (
                    f"BiMap layer {i} has zero gradient"
                )
            elif isinstance(layer, (BatchNormSPDMean, BatchNormSPDMeanScalarVariance)):
                assert layer.parametrizations.Covbias.original.grad is not None, (
                    f"BatchNormSPDMean layer {i} Covbias has no gradient"
                )
                assert layer.parametrizations.Covbias.original.grad.abs().sum() > 0, (
                    f"BatchNormSPDMean layer {i} Covbias has zero gradient"
                )
            elif isinstance(layer, BatchNormSPDMeanScalarVariance):
                assert layer.parametrizations.stdScalarbias.original.grad is not None, (
                    f"BatchNormSPDMeanScalarVariance layer {i} stdScalarbias has no gradient"
                )
                assert (
                    layer.parametrizations.stdScalarbias.original.grad.abs().sum() > 0
                ), f"BatchNormSPDMeanScalarVariance layer {i} has zero gradient"

    @pytest.mark.parametrize("invalid_vec_type", ["invalid", "vector", ""])
    def test_invalid_vec_type(self, invalid_vec_type, device, dtype, generator):
        """Test that invalid vec_type raises AssertionError"""
        with pytest.raises(AssertionError, match="vec_type must be"):
            SPDnet(
                input_dim=10,
                hidden_layers_size=[8],
                output_dim=3,
                vec_type=invalid_vec_type,
                device=device,
                dtype=dtype,
                generator=generator,
            )

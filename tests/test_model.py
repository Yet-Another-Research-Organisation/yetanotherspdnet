import warnings

import pytest
import torch
from torch.nn.utils import parametrize
from torch.testing import assert_close

# from utils import is_symmetric
from yetanotherspdnet.model import (
    GBWBNPlacementWarning,
    GBWBNRResNet,
    RResNet,
    SPDnet,
)
from yetanotherspdnet.nn.base import BiMap, LogEig, ReEig, Vec, Vech
from yetanotherspdnet.nn.batchnorm import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
)
from yetanotherspdnet.random.spd import random_SPD


@pytest.fixture(scope="module")
def dtype():
    return torch.float32


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
        assert torch.allclose(
            last_tensor, last_tensor.transpose(-1, -2), atol=1e-5, rtol=1e-5
        )

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

    def test_dynamic_parametrization(self, device, dtype, generator):
        """
        Test that dynamic parametrization is set up correctly
        """
        model = SPDnet(
            input_dim=10,
            hidden_layers_size=[8, 6],
            output_dim=3,
            bimap_parametrization_mode="dynamic",
            bimap_n_steps_ref_update=2,
            batchnorm=True,
            batchnorm_parametrization_mode="dynamic",
            batchnorm_n_steps_ref_update=2,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.register_optimizer_hook(optimizer)

        for module in model.modules():
            if isinstance(
                module, (BiMap, BatchNormSPDMean, BatchNormSPDMeanScalarVariance)
            ):
                assert hasattr(module, "is_dynamic") and module.is_dynamic is True

        # first time going through the model
        X1 = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        # Forward + backward + update
        optimizer.zero_grad()
        output = model(X1)
        # check current_ref_step is correct
        for module in model.modules():
            if isinstance(
                module, (BiMap, BatchNormSPDMean, BatchNormSPDMeanScalarVariance)
            ):
                assert module.current_ref_step == 1
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # check that reference_point did not change
        for module in model.modules():
            if isinstance(module, BiMap):
                assert_close(
                    module.stiefel_parametrization.reference_point,
                    module.stiefel_parametrization.initial_reference,
                )
            if isinstance(module, (BatchNormSPDMean, BatchNormSPDMeanScalarVariance)):
                assert_close(
                    module.spd_parametrization.reference_point,
                    module.spd_parametrization.initial_reference,
                )

        # second time going through the model
        X2 = random_SPD(10, 5, device=device, dtype=dtype, generator=generator)
        # Forward + backward + update
        optimizer.zero_grad()
        output = model(X2)
        # check current_ref_step is correct
        for module in model.modules():
            if isinstance(
                module, (BiMap, BatchNormSPDMean, BatchNormSPDMeanScalarVariance)
            ):
                assert module.current_ref_step == 2
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # check that reference_point changed, tangent vector and current_ref_step re-initialized
        for module in model.modules():
            if isinstance(module, BiMap):
                assert not torch.allclose(
                    module.stiefel_parametrization.reference_point,
                    module.stiefel_parametrization.initial_reference,
                )
                assert_close(
                    module.parametrizations.weight.original,
                    torch.zeros(
                        (module.n_in, module.n_out), device=device, dtype=dtype
                    ),
                )
                assert module.current_ref_step == 0
            if isinstance(module, (BatchNormSPDMean, BatchNormSPDMeanScalarVariance)):
                assert not torch.allclose(
                    module.spd_parametrization.reference_point,
                    module.spd_parametrization.initial_reference,
                )
                assert_close(
                    module.parametrizations.Covbias.original,
                    torch.zeros(
                        (module.n_features, module.n_features),
                        device=device,
                        dtype=dtype,
                    ),
                )
                assert module.current_ref_step == 0

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
            if isinstance(layer, BatchNormSPDMeanScalarVariance):
                assert layer.parametrizations.stdScalarbias.original.grad is not None, (
                    f"BatchNormSPDMeanScalarVariance layer {i} stdScalarbias has no gradient"
                )
                assert (
                    layer.parametrizations.stdScalarbias.original.grad.abs().sum() > 0
                ), f"BatchNormSPDMeanScalarVariance layer {i} has zero gradient"

    @pytest.mark.parametrize("invalid_vec_type", ["invalid", "vector", ""])
    def test_invalid_vec_type(self, invalid_vec_type, device, dtype, generator):
        """Test that invalid vec_type raises ValueError"""
        with pytest.raises(ValueError, match="vec_type must be"):
            SPDnet(
                input_dim=10,
                hidden_layers_size=[8],
                output_dim=3,
                vec_type=invalid_vec_type,
                device=device,
                dtype=dtype,
                generator=generator,
            )


class TestGBWBNRResNet:
    """Test suite for GBWBNRResNet model."""

    # RResNet models use nested eigenvalue decompositions (exp map) that
    # require float64 for numerical stability.

    @pytest.mark.parametrize(
        "input_dim, hidden_dim, output_dim",
        [(10, 5, 3), (20, 8, 5)],
    )
    def test_initialization(self, input_dim, hidden_dim, output_dim, device, generator):
        """Test GBWBNRResNet initializes correctly."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        assert model.input_dim == input_dim
        assert model.hidden_dim == hidden_dim
        assert model.output_dim == output_dim

    @pytest.mark.parametrize("n_samples", [1, 10])
    def test_forward_shape(self, n_samples, device, generator):
        """Forward pass returns correct shape."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(
            10, n_samples, device=device, dtype=torch.float64, generator=generator
        )
        if n_samples == 1:
            # a single matrix has no batch statistics: BatchNorm falls back to
            # its running statistics instead of collapsing the input to identity
            with pytest.warns(UserWarning, match="single matrix"):
                output = model(X)
        else:
            output = model(X)
        expected_shape = (n_samples, 3) if n_samples > 1 else (3,)
        assert output.shape == expected_shape
        assert output.dtype == torch.float64

    @pytest.mark.parametrize("vec_type", ["vec", "vech"])
    def test_vec_type(self, vec_type, device, generator):
        """Both vectorization types work."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            vec_type=vec_type,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    def test_backward(self, device, generator):
        """Backward pass produces finite gradients."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            batchnorm_mean_type="affine_invariant",
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        loss = output.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_softmax(self, device, generator):
        """Softmax output sums to 1."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            softmax=True,
            batchnorm_mean_type="affine_invariant",
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert_close(
            output.sum(dim=-1),
            torch.ones(4, dtype=torch.float64, device=device),
        )

    @pytest.mark.parametrize("batchnorm", [True, False])
    def test_with_without_batchnorm(self, batchnorm, device, generator):
        """Works with and without batchnorm."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            batchnorm=batchnorm,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    @pytest.mark.parametrize(
        "batchnorm_mean_type",
        ["bures_wasserstein", "affine_invariant", "log_euclidean"],
    )
    def test_batchnorm_mean_types(self, batchnorm_mean_type, device, generator):
        """Different batchnorm mean types work."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            batchnorm=True,
            batchnorm_mean_type=batchnorm_mean_type,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_spectrum_types(self, spectrum_type, device, generator):
        """Both spectrum types work."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            spectrum_type=spectrum_type,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    def test_get_last_tensor(self, device, generator):
        """get_last_tensor returns SPD tensor before vectorization."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        last = model.get_last_tensor(X)
        assert last.shape == (4, 5, 5)

    def test_repr_and_layers_str(self, device, generator):
        """repr and layers_str return informative strings."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        r = repr(model)
        assert "GBWBNRResNet" in r
        assert "input_dim=10" in r
        ls = model.layers_str()
        assert "Layers:" in ls

    def test_model_hash(self, device, generator):
        """Model hash is consistent."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        h1 = model.create_model_name_hash()
        h2 = model.get_model_hash()
        assert h1 == h2
        assert len(h1) == 8

    def test_train_eval_modes(self, device, generator):
        """Train and eval modes produce valid output."""
        from yetanotherspdnet.model import GBWBNRResNet

        model = GBWBNRResNet(
            input_dim=10,
            hidden_dim=5,
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)

        model.train()
        out_train = model(X)
        assert out_train.shape == (4, 3)

        model.eval()
        out_eval = model(X)
        assert out_eval.shape == (4, 3)

    @pytest.mark.parametrize("invalid_vec_type", ["invalid", "matric", ""])
    def test_invalid_vec_type(self, invalid_vec_type, device, generator):
        """Invalid vec_type raises."""
        from yetanotherspdnet.model import GBWBNRResNet

        with pytest.raises(ValueError):
            GBWBNRResNet(
                input_dim=10,
                hidden_dim=5,
                output_dim=3,
                vec_type=invalid_vec_type,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )


class TestRResNet:
    """Test suite for RResNet model."""

    # RResNet models use nested eigenvalue decompositions (exp map) that
    # require float64 for numerical stability.

    @pytest.mark.parametrize(
        "input_dim, hidden_layers, n_residual, output_dim",
        [
            (10, [5], [1], 3),
            (20, [15, 10], [1, 2], 5),
        ],
    )
    def test_initialization(
        self, input_dim, hidden_layers, n_residual, output_dim, device, generator
    ):
        """Test RResNet initializes correctly."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=input_dim,
            hidden_layers_size=hidden_layers,
            n_residual_blocks=n_residual,
            output_dim=output_dim,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        assert model.input_dim == input_dim
        assert model.hidden_layers_size == hidden_layers
        assert model.n_residual_blocks == n_residual
        assert model.output_dim == output_dim

    @pytest.mark.parametrize("n_samples", [1, 10])
    def test_forward_shape(self, n_samples, device, generator):
        """Forward pass returns correct shape."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(
            10, n_samples, device=device, dtype=torch.float64, generator=generator
        )
        output = model(X)
        expected_shape = (n_samples, 3) if n_samples > 1 else (3,)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("vec_type", ["vec", "vech"])
    def test_vec_type(self, vec_type, device, generator):
        """Both vectorization types work."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            vec_type=vec_type,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    def test_backward(self, device, generator):
        """Backward pass produces finite gradients."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        loss = output.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_multi_stage(self, device, generator):
        """Multi-stage architecture works end-to-end."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=15,
            hidden_layers_size=[10, 5],
            n_residual_blocks=[1, 2],
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(15, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)
        output.sum().backward()

    @pytest.mark.parametrize("reeig", [True, False])
    def test_with_reeig(self, reeig, device, generator):
        """Works with and without ReEig."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            reeig=reeig,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    @pytest.mark.parametrize("batchnorm", [True, False])
    def test_with_without_batchnorm(self, batchnorm, device, generator):
        """Works with and without batchnorm."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            batchnorm=batchnorm,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    def test_zero_residual_blocks(self, device, generator):
        """Stage with 0 residual blocks (just BiMap) works."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[0],
            output_dim=3,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert output.shape == (4, 3)

    def test_softmax(self, device, generator):
        """Softmax output sums to 1."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            softmax=True,
            use_autograd=True,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        X = random_SPD(10, 4, device=device, dtype=torch.float64, generator=generator)
        output = model(X)
        assert_close(
            output.sum(dim=-1),
            torch.ones(4, dtype=torch.float64, device=device),
        )

    def test_mismatched_lengths_raises(self, device, generator):
        """Mismatched hidden_layers_size and n_residual_blocks raises."""
        from yetanotherspdnet.model import RResNet

        with pytest.raises(ValueError, match="same length"):
            RResNet(
                input_dim=10,
                hidden_layers_size=[5, 3],
                n_residual_blocks=[1],
                output_dim=3,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )

    def test_repr_and_layers_str(self, device, generator):
        """repr and layers_str return informative strings."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[8, 5],
            n_residual_blocks=[1, 2],
            output_dim=3,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        r = repr(model)
        assert "RResNet" in r
        assert "input_dim=10" in r
        ls = model.layers_str()
        assert "Layers:" in ls

    def test_model_hash(self, device, generator):
        """Model hash is consistent."""
        from yetanotherspdnet.model import RResNet

        model = RResNet(
            input_dim=10,
            hidden_layers_size=[5],
            n_residual_blocks=[1],
            output_dim=3,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        h1 = model.create_model_name_hash()
        h2 = model.get_model_hash()
        assert h1 == h2

    @pytest.mark.parametrize("invalid_vec_type", ["invalid", "matric", ""])
    def test_invalid_vec_type(self, invalid_vec_type, device, generator):
        """Invalid vec_type raises."""
        from yetanotherspdnet.model import RResNet

        with pytest.raises(ValueError):
            RResNet(
                input_dim=10,
                hidden_layers_size=[5],
                n_residual_blocks=[1],
                output_dim=3,
                vec_type=invalid_vec_type,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )


# Model hashes are used to name experiment runs: they must not change across
# refactors. Values computed with the release that made them stable.
_MODEL_HASHES = [
    (
        SPDnet,
        {"input_dim": 8, "hidden_layers_size": [6, 4], "output_dim": 3},
        "f8ffbe94",
    ),
    (
        SPDnet,
        {
            "input_dim": 8,
            "hidden_layers_size": [6, 4],
            "output_dim": 3,
            "batchnorm": True,
            "batchnorm_type": "mean_var_scalar",
            "batchnorm_mean_type": "bures_wasserstein",
            # the placement of GBWBN before batchnorm_position existed (the
            # default is now GBWBN -> ReEig, which changes the hash)
            "batchnorm_position": "after_reeig",
            "use_autograd": {"bimap": True},
        },
        "5a2bd627",
    ),
    (
        SPDnet,
        {
            "input_dim": 8,
            "hidden_layers_size": [6],
            "output_dim": 3,
            "batchnorm": True,
            "batchnorm_norm_and_bias_mode": "parallel_transport",
            "vec_type": "vech",
        },
        "52594341",
    ),
    (
        RResNet,
        {
            "input_dim": 8,
            "hidden_layers_size": [6],
            "n_residual_blocks": [1],
            "output_dim": 3,
        },
        "b90d1c26",
    ),
    (
        RResNet,
        {
            "input_dim": 8,
            "hidden_layers_size": [6, 4],
            "n_residual_blocks": [1, 2],
            "output_dim": 3,
            "batchnorm": True,
            "residual_metric": "log_euclidean",
        },
        "ed8eb3c6",
    ),
    # SPDnet + GBWBN: "after_reeig" is the placement (and the hash) before
    # batchnorm_position existed; the default is now GBWBN -> ReEig
    *[
        (
            SPDnet,
            {
                "input_dim": 8,
                "hidden_layers_size": [6],
                "output_dim": 3,
                "batchnorm": True,
                "batchnorm_type": "mean_var_scalar",
                "batchnorm_mean_type": "bures_wasserstein",
                **position,
            },
            expected,
        )
        for position, expected in [
            ({}, "5d81eefe"),
            ({"batchnorm_position": "before_reeig"}, "5d81eefe"),
            ({"batchnorm_position": "after_reeig"}, "475b16dd"),
        ]
    ],
    (GBWBNRResNet, {"input_dim": 8, "hidden_dim": 6, "output_dim": 3}, "29deb207"),
    (
        GBWBNRResNet,
        {
            "input_dim": 8,
            "hidden_dim": 6,
            "output_dim": 3,
            "batchnorm_bw_options": {"bw_theta": 1.0},
            "use_autograd": True,
        },
        "7f6c0d50",
    ),
]


@pytest.mark.filterwarnings("ignore::yetanotherspdnet.model.GBWBNPlacementWarning")
@pytest.mark.parametrize("model, kwargs, expected", _MODEL_HASHES)
def test_model_hash_is_stable(model, kwargs, expected):
    net = model(**kwargs, dtype=torch.float64, generator=torch.Generator())
    assert net.get_model_hash() == expected


@pytest.mark.filterwarnings("ignore::yetanotherspdnet.model.GBWBNPlacementWarning")
@pytest.mark.parametrize("model, kwargs, expected", _MODEL_HASHES)
def test_use_autograd_expansion(model, kwargs, expected):
    net = model(**kwargs, dtype=torch.float64, generator=torch.Generator())
    use_autograd = kwargs.get("use_autograd", False)
    assert set(net.use_autograd) == set(model._use_autograd_keys)
    for key, value in net.use_autograd.items():
        if isinstance(use_autograd, bool):
            assert value is use_autograd
        else:
            assert value is use_autograd.get(key, False)


# ---------------------------------------------------------------------------
# GBWBN after the ReEig: one GBWBNPlacementWarning per model, no hash change
# ---------------------------------------------------------------------------
_GBWBN = {
    "batchnorm": True,
    "batchnorm_type": "mean_var_scalar",
    "batchnorm_mean_type": "bures_wasserstein",
}


def _placement_warnings(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build()
    found = [w for w in caught if issubclass(w.category, GBWBNPlacementWarning)]
    return model, found


def test_gbwbn_after_reeig_warns_once_per_model():
    # two GBWBN layers, one warning per construction, pointing at the caller
    for _ in range(2):
        _, found = _placement_warnings(
            lambda: SPDnet(8, [6, 4], 3, **_GBWBN, batchnorm_position="after_reeig")
        )
        assert len(found) == 1
        assert found[0].filename == __file__
        message = str(found[0].message)
        assert "before_reeig" in message
        assert "user_guide/batchnorm.html#" in message
    with pytest.warns(GBWBNPlacementWarning, match="leave its domain"):
        SPDnet(8, [6], 3, **_GBWBN, batchnorm_position="after_reeig")


def test_gbwbn_placement_warning_keeps_the_hash():
    # the hash of the former placement (see _MODEL_HASHES)
    with pytest.warns(GBWBNPlacementWarning):
        model = SPDnet(
            8,
            [6],
            3,
            **_GBWBN,
            batchnorm_position="after_reeig",
            dtype=torch.float64,
            generator=torch.Generator(),
        )
    assert model.get_model_hash() == "475b16dd"


@pytest.mark.parametrize(
    "build",
    [
        # the default GBWBN placement, and the explicit one
        lambda: SPDnet(8, [6, 4], 3, **_GBWBN),
        lambda: SPDnet(8, [6, 4], 3, **_GBWBN, batchnorm_position="before_reeig"),
        # the other batch norms, in both placements
        lambda: SPDnet(8, [6, 4], 3, batchnorm=True),
        lambda: SPDnet(8, [6, 4], 3, batchnorm=True, batchnorm_position="after_reeig"),
        lambda: SPDnet(
            8,
            [6],
            3,
            batchnorm=True,
            batchnorm_type="mean_var_scalar",
            batchnorm_mean_type="affine_invariant",
            batchnorm_position="after_reeig",
        ),
        lambda: SPDnet(
            8,
            [6],
            3,
            batchnorm=True,
            batchnorm_mean_type="bures_wasserstein",
            batchnorm_position="after_reeig",
        ),
        # GBWBN options without batch norm
        lambda: SPDnet(
            8,
            [6],
            3,
            **(_GBWBN | {"batchnorm": False}),
            batchnorm_position="after_reeig",
        ),
        # the residual networks floor the GBWBN output in their residual block
        lambda: GBWBNRResNet(8, 6, 3),
        lambda: RResNet(8, [6, 4], [1, 1], 3, **_GBWBN),
        lambda: RResNet(8, [6, 4], [1, 1], 3, **_GBWBN, reeig=True),
    ],
)
def test_no_gbwbn_placement_warning(build):
    _, found = _placement_warnings(build)
    assert found == []

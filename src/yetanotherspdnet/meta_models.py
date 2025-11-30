#!/usr/bin/env python3
"""
Meta Models - Unified Model Creation Interface
==============================================

Provides a unified interface for creating different deep learning models.
All models are based on SPDNet architecture with different backbones.

Available Models:
- SPDNet: Pure SPDNet without CNN backbone
- CNNSPDNet: Custom CNN backbone + SPDNet
- ResNetSPDNet: ResNet backbone + SPDNet (ResNet18/34/50/101/152)
- MobileNetV3LargeSPDNet: MobileNetV3 backbone + SPDNet
- EfficientNetB0SPDNet: EfficientNet-B0 backbone + SPDNet
- EfficientNetB4SPDNet: EfficientNet-B4 backbone + SPDNet
- CNN: Pure CNN without SPDNet (baseline)

Date: 2025-10-19
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any


def create_model(
    model_name: str,
    output_dim: int,
    input_channels: int = 3,
    device: Optional[torch.device] = None,
    precision: Optional[torch.dtype] = torch.float32,
    seed: Optional[int] = None,
    verbose: bool = False,
    **model_kwargs
) -> nn.Module:
    """
    Create a model from available architectures.
    
    Parameters
    ----------
    model_name : str
        Name of the model architecture. Options:
        - 'spdnet': Pure SPDNet
        - 'cnnspdnet': Custom CNN + SPDNet
        - 'resnetspdnet': ResNet + SPDNet
        - 'mobilenetspdnet': MobileNetV3Large + SPDNet
        - 'efficientnetb0spdnet': EfficientNet-B0 + SPDNet
        - 'efficientnetb4spdnet': EfficientNet-B4 + SPDNet
        - 'cnn': Pure CNN (baseline)
        
    output_dim : int
        Number of output classes
        
    input_channels : int, default=3
        Number of input channels (1 for grayscale, 3 for RGB, 4 for complex data)
        
    device : torch.device, optional
        Device to place model on (default: auto-detect CUDA)
        
    precision : torch.dtype, default=torch.float32
        Model precision (torch.float32 or torch.float64)
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool, default=False
        Print model information
        
    **model_kwargs : dict
        Model-specific parameters
        
    Model-Specific Parameters
    -------------------------
    
    SPDNet:
        - input_dim: int - Input SPD matrix dimension
        - hidden_layers_size: List[int] - Hidden layer dimensions
        - eps: float - Regularization for ReEig (default: 1e-3)
        - batchnorm: bool - Use SPD batchnorm (default: False)
        - batchnorm_method: str - SPD mean method (default: 'geometric_arithmetic_harmonic')
        - softmax: bool - Apply softmax to output (default: False)
        - use_autograd: bool - Use autograd (default: False)
        - dropout_rate: float - Dropout rate (default: 0.0)
        - use_vech: bool - Use vech vectorization (default: False)
        
    CNNSPDNet:
        - feature_channels: List[int] - CNN channel dimensions (default: [64, 192])
        - hidden_layers_size: List[int] - SPDNet hidden dimensions (default: [24])
        - shrinkage: str - Covariance shrinkage method (default: 'ledoit_wolf')
        - lbda: str or float - Shrinkage parameter (default: 'optimal')
        - eps: float - Regularization (default: 1e-6)
        - batchnorm: bool - Use SPD batchnorm (default: False)
        - dropout_rate: float - Dropout rate (default: 0.0)
        
    ResNetSPDNet:
        - resnet_type: str - ResNet variant (default: 'resnet18')
          Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        - num_layers: int - Number of ResNet blocks to use (default: 2)
        - hidden_layers_size: List[int] - SPDNet hidden dimensions
        - pretrained: bool - Use ImageNet pretrained weights (default: True)
        - shrinkage: str - Covariance shrinkage (default: 'ledoit_wolf')
        - lbda: str or float - Shrinkage parameter (default: 'optimal')
        
    MobileNetV3LargeSPDNet:
        - num_layers: int - Number of MobileNet blocks (default: 5)
        - hidden_layers_size: List[int] - SPDNet hidden dimensions
        - pretrained: bool - Use ImageNet pretrained weights (default: True)
        - shrinkage: str - Covariance shrinkage (default: 'ledoit_wolf')
        
    EfficientNetB0SPDNet / EfficientNetB4SPDNet:
        - num_layers: int - Number of EfficientNet blocks (default: 5)
        - hidden_layers_size: List[int] - SPDNet hidden dimensions
        - pretrained: bool - Use ImageNet pretrained weights (default: True)
        - shrinkage: str - Covariance shrinkage (default: 'ledoit_wolf')
        
    CNN:
        - feature_channels: List[int] - CNN channel dimensions (default: [64, 128, 256])
        - dropout_rate: float - Dropout rate (default: 0.3)
    
    Returns
    -------
    nn.Module
        Initialized PyTorch model
        
    Examples
    --------
    >>> # Pure SPDNet for covariance matrices
    >>> model = create_model('spdnet', output_dim=10, input_dim=50)
    
    >>> # CNN + SPDNet for SAR images
    >>> model = create_model('cnnspdnet', output_dim=7, input_channels=1,
    ...                      feature_channels=[64, 192], hidden_layers_size=[24])
    
    >>> # ResNet18 + SPDNet with pretrained backbone
    >>> model = create_model('resnetspdnet', output_dim=10, input_channels=3,
    ...                      resnet_type='resnet18', pretrained=True)
    
    >>> # MobileNet for mobile deployment
    >>> model = create_model('mobilenetspdnet', output_dim=4, pretrained=True)
    """
    
    # Normalize model name
    model_name = model_name.lower().replace('-', '').replace('_', '')
    
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating Model: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"Output dimension: {output_dim}")
        print(f"Input channels: {input_channels}")
        print(f"Device: {device}")
        print(f"Precision: {precision}")
        print(f"Seed: {seed}")
    
    # Route to appropriate model
    if model_name in ['spdnet']:
        return _create_spdnet(
            output_dim, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['cnnspdnet', 'cnn_spdnet']:
        return _create_cnnspdnet(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['resnetspdnet', 'resnet_spdnet', 'resnet']:
        return _create_resnetspdnet(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['mobilenetspdnet', 'mobilenet_spdnet', 'mobilenetv3', 'mobilenet']:
        return _create_mobilenetspdnet(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['efficientnetb0spdnet', 'efficientnetb0', 'effb0']:
        return _create_efficientnetb0spdnet(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['efficientnetb4spdnet', 'efficientnetb4', 'effb4']:
        return _create_efficientnetb4spdnet(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    elif model_name in ['cnn']:
        return _create_cnn(
            output_dim, input_channels, device, precision, seed, verbose, **model_kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: "
            f"'spdnet', 'cnnspdnet', 'resnetspdnet', 'mobilenetspdnet', "
            f"'efficientnetb0spdnet', 'efficientnetb4spdnet', 'cnn'"
        )


def _create_spdnet(
    output_dim: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create pure SPDNet model"""
    from yetanotherspdnet.models import SPDnet
    
    # Extract SPDNet-specific parameters
    input_dim = kwargs.get('input_dim', 50)
    hidden_layers_size = kwargs.get('hidden_layers_size', [30, 20])
    reeig_eps = kwargs.get('eps', 1e-3)
    batchnorm = kwargs.get('batchnorm', False)
    
    # Map old parameter names to new ones
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    batchnorm_mean_type = batchnorm_method  # New name in SPDnet
    
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', False)
    use_vech = kwargs.get('use_vech', False)
    use_logeig = kwargs.get('use_logeig', True)
    
    # Map vec_type parameter
    vec_type = "vech" if use_vech else "vec"
    
    # Batchnorm parameters
    batchnorm_momentum = kwargs.get('momentum', 0.01)
    batchnorm_mean_options = kwargs.get('batchnorm_mean_options', None)
    # If None, will fallback to batchnorm_mean_type in SPDnet
    batchnorm_adaptive_mean_type = kwargs.get('batchnorm_adaptive_mean_type', None)
    
    # Bimap parametrization parameters
    bimap_parametrized = kwargs.get('bimap_parametrized', True)
    bimap_parametrization = kwargs.get('bimap_parametrization', None)
    bimap_parametrization_name = kwargs.get('bimap_parametrization_name', None)
    
    # Handle parametrization by name or direct object
    if bimap_parametrization_name is not None:
        if bimap_parametrization_name == 'orthogonal':
            from torch.nn.utils import parametrizations
            bimap_parametrization = parametrizations.orthogonal
        elif bimap_parametrization_name == 'StiefelProjectionQRParametrization':
            from yetanotherspdnet.nn.base import StiefelProjectionQRParametrization
            bimap_parametrization = StiefelProjectionQRParametrization
        else:
            raise ValueError(f"Unknown parametrization name: {bimap_parametrization_name}")
    elif bimap_parametrization is None and bimap_parametrized:
        # Default to orthogonal parametrization
        from torch.nn.utils import parametrizations
        bimap_parametrization = parametrizations.orthogonal
    bimap_parametrization_options = kwargs.get('bimap_parametrization_options', None)
    
    # Create generator if seed is provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    
    model = SPDnet(
        input_dim=input_dim,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        softmax=softmax,
        reeig_eps=reeig_eps,
        bimap_parametrized=bimap_parametrized,
        bimap_parametrization=bimap_parametrization,
        bimap_parametrization_options=bimap_parametrization_options,
        batchnorm=batchnorm,
        batchnorm_mean_type=batchnorm_mean_type,
        batchnorm_mean_options=batchnorm_mean_options,
        batchnorm_adaptive_mean_type=batchnorm_adaptive_mean_type,
        batchnorm_momentum=batchnorm_momentum,
        vec_type=vec_type,
        use_logeig=use_logeig,
        device=device,
        dtype=precision,
        generator=generator,
        use_autograd=use_autograd,
    )
    
    if verbose:
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden layers: {hidden_layers_size}")
        print(f"  Batchnorm: {batchnorm}")
        print(f"  Vec type: {vec_type}")
    
    return model


def _create_cnnspdnet(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create CNN + SPDNet model"""
    from src.models import CNNSPDNet
    
    # Extract CNNSPDNet-specific parameters
    feature_channels = kwargs.get('feature_channels', [64, 192])
    hidden_layers_size = kwargs.get('hidden_layers_size', [24])
    shrinkage = kwargs.get('shrinkage', 'ledoit_wolf')
    lbda = kwargs.get('lbda', 'optimal')
    eps = kwargs.get('eps', 1e-6)
    batchnorm = kwargs.get('batchnorm', False)
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', True)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    use_vech = kwargs.get('use_vech', False)
    momentum = kwargs.get('momentum', 0.01)
    max_iter_batchnorm = kwargs.get('max_iter_batchnorm', 10)
    orthogonal_map = kwargs.get('orthogonal_map', None)
    batchnorm_enforce_float64 = kwargs.get('batchnorm_enforce_float64', True)
    
    model = CNNSPDNet(
        input_channels=input_channels,
        feature_channels=feature_channels,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        shrinkage=shrinkage,
        lbda=lbda,
        eps=eps,
        batchnorm=batchnorm,
        batchnorm_method=batchnorm_method,
        device=device,
        precision=precision,
        softmax=softmax,
        seed=seed,
        use_autograd=use_autograd,
        dropout_rate=dropout_rate,
        use_vech=use_vech,
        momentum=momentum,
        max_iter_batchnorm=max_iter_batchnorm,
        orthogonal_map=orthogonal_map,
        batchnorm_enforce_float64=batchnorm_enforce_float64,
    )
    
    if verbose:
        print(f"  Feature channels: {feature_channels}")
        print(f"  SPDNet hidden layers: {hidden_layers_size}")
        print(f"  Shrinkage: {shrinkage}")
    
    return model


def _create_resnetspdnet(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create ResNet + SPDNet model"""
    from src.models import ResNetSPDNet
    
    # Extract ResNetSPDNet-specific parameters
    resnet_type = kwargs.get('resnet_type', 'resnet18')
    num_layers = kwargs.get('num_layers', 2)
    hidden_layers_size = kwargs.get('hidden_layers_size', None)
    pretrained = kwargs.get('pretrained', True)
    shrinkage = kwargs.get('shrinkage', 'ledoit_wolf')
    lbda = kwargs.get('lbda', 'optimal')
    eps = kwargs.get('eps', 1e-3)
    batchnorm = kwargs.get('batchnorm', False)
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', False)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    use_vech = kwargs.get('use_vech', False)
    use_chunking = kwargs.get('use_chunking', False)
    chunk_size = kwargs.get('chunk_size', None)
    momentum = kwargs.get('momentum', 0.01)
    max_iter_batchnorm = kwargs.get('max_iter_batchnorm', 10)
    orthogonal_map = kwargs.get('orthogonal_map', None)
    batchnorm_enforce_float64 = kwargs.get('batchnorm_enforce_float64', True)
    
    model = ResNetSPDNet(
        resnet_type=resnet_type,
        num_layers=num_layers,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        pretrained=pretrained,
        input_channels=input_channels,
        shrinkage=shrinkage,
        lbda=lbda,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        softmax=softmax,
        eps=eps,
        batchnorm=batchnorm,
        batchnorm_method=batchnorm_method,
        seed=seed,
        device=device,
        precision=precision,
        use_autograd=use_autograd,
        dropout_rate=dropout_rate,
        use_vech=use_vech,
        momentum=momentum,
        max_iter_batchnorm=max_iter_batchnorm,
        orthogonal_map=orthogonal_map,
        batchnorm_enforce_float64=batchnorm_enforce_float64,
    )
    
    if verbose:
        print(f"  ResNet type: {resnet_type}")
        print(f"  Num layers: {num_layers}")
        print(f"  Pretrained: {pretrained}")
        print(f"  SPDNet hidden layers: {hidden_layers_size}")
    
    return model


def _create_mobilenetspdnet(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create MobileNetV3Large + SPDNet model"""
    from src.models import MobileNetV3LargeSPDNet
    
    # Extract parameters
    num_layers = kwargs.get('num_layers', 5)
    hidden_layers_size = kwargs.get('hidden_layers_size', None)
    pretrained = kwargs.get('pretrained', True)
    shrinkage = kwargs.get('shrinkage', 'ledoit_wolf')
    lbda = kwargs.get('lbda', 'optimal')
    eps = kwargs.get('eps', 1e-3)
    batchnorm = kwargs.get('batchnorm', False)
    use_chunking = kwargs.get('use_chunking', False)
    chunk_size = kwargs.get('chunk_size', None)
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', False)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    use_vech = kwargs.get('use_vech', False)
    momentum = kwargs.get('momentum', 0.01)
    max_iter_batchnorm = kwargs.get('max_iter_batchnorm', 10)
    orthogonal_map = kwargs.get('orthogonal_map', None)
    batchnorm_enforce_float64 = kwargs.get('batchnorm_enforce_float64', True)
    
    model = MobileNetV3LargeSPDNet(
        num_layers=num_layers,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        pretrained=pretrained,
        input_channels=input_channels,
        shrinkage=shrinkage,
        lbda=lbda,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        eps=eps,
        batchnorm=batchnorm,
        batchnorm_method=batchnorm_method,
        seed=seed,
        device=device,
        precision=precision,
        softmax=softmax,
        use_autograd=use_autograd,
        dropout_rate=dropout_rate,
        use_vech=use_vech,
        momentum=momentum,
        max_iter_batchnorm=max_iter_batchnorm,
        orthogonal_map=orthogonal_map,
        batchnorm_enforce_float64=batchnorm_enforce_float64,
    )
    
    if verbose:
        print(f"  Num layers: {num_layers}")
        print(f"  Pretrained: {pretrained}")
    
    return model


def _create_efficientnetb0spdnet(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create EfficientNet-B0 + SPDNet model"""
    from src.models import EfficientNetB0SPDNet
    
    # Extract parameters
    num_layers = kwargs.get('num_layers', 5)
    hidden_layers_size = kwargs.get('hidden_layers_size', None)
    pretrained = kwargs.get('pretrained', True)
    shrinkage = kwargs.get('shrinkage', 'ledoit_wolf')
    lbda = kwargs.get('lbda', 'optimal')
    eps = kwargs.get('eps', 1e-3)
    batchnorm = kwargs.get('batchnorm', False)
    use_chunking = kwargs.get('use_chunking', False)
    chunk_size = kwargs.get('chunk_size', None)
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', False)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    use_vech = kwargs.get('use_vech', False)
    momentum = kwargs.get('momentum', 0.01)
    max_iter_batchnorm = kwargs.get('max_iter_batchnorm', 10)
    orthogonal_map = kwargs.get('orthogonal_map', None)
    batchnorm_enforce_float64 = kwargs.get('batchnorm_enforce_float64', True)
    
    model = EfficientNetB0SPDNet(
        num_layers=num_layers,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        pretrained=pretrained,
        input_channels=input_channels,
        shrinkage=shrinkage,
        lbda=lbda,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        eps=eps,
        batchnorm=batchnorm,
        batchnorm_method=batchnorm_method,
        seed=seed,
        device=device,
        precision=precision,
        softmax=softmax,
        use_autograd=use_autograd,
        dropout_rate=dropout_rate,
        use_vech=use_vech,
        momentum=momentum,
        max_iter_batchnorm=max_iter_batchnorm,
        orthogonal_map=orthogonal_map,
        batchnorm_enforce_float64=batchnorm_enforce_float64,
    )
    
    if verbose:
        print(f"  Num layers: {num_layers}")
        print(f"  Pretrained: {pretrained}")
    
    return model


def _create_efficientnetb4spdnet(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create EfficientNet-B4 + SPDNet model"""
    from src.models import EfficientNetB4SPDNet
    
    # Extract parameters
    num_layers = kwargs.get('num_layers', 5)
    hidden_layers_size = kwargs.get('hidden_layers_size', None)
    pretrained = kwargs.get('pretrained', True)
    shrinkage = kwargs.get('shrinkage', 'ledoit_wolf')
    lbda = kwargs.get('lbda', 'optimal')
    eps = kwargs.get('eps', 1e-3)
    batchnorm = kwargs.get('batchnorm', False)
    use_chunking = kwargs.get('use_chunking', False)
    chunk_size = kwargs.get('chunk_size', None)
    batchnorm_method = kwargs.get('batchnorm_method', 'geometric_arithmetic_harmonic')
    softmax = kwargs.get('softmax', False)
    use_autograd = kwargs.get('use_autograd', False)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    use_vech = kwargs.get('use_vech', False)
    momentum = kwargs.get('momentum', 0.01)
    max_iter_batchnorm = kwargs.get('max_iter_batchnorm', 10)
    orthogonal_map = kwargs.get('orthogonal_map', None)
    batchnorm_enforce_float64 = kwargs.get('batchnorm_enforce_float64', True)
    
    model = EfficientNetB4SPDNet(
        num_layers=num_layers,
        hidden_layers_size=hidden_layers_size,
        output_dim=output_dim,
        pretrained=pretrained,
        input_channels=input_channels,
        shrinkage=shrinkage,
        lbda=lbda,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        eps=eps,
        batchnorm=batchnorm,
        batchnorm_method=batchnorm_method,
        seed=seed,
        device=device,
        precision=precision,
        softmax=softmax,
        use_autograd=use_autograd,
        dropout_rate=dropout_rate,
        use_vech=use_vech,
        momentum=momentum,
        max_iter_batchnorm=max_iter_batchnorm,
        orthogonal_map=orthogonal_map,
        batchnorm_enforce_float64=batchnorm_enforce_float64,
    )
    
    if verbose:
        print(f"  Num layers: {num_layers}")
        print(f"  Pretrained: {pretrained}")
    
    return model


def _create_cnn(
    output_dim: int,
    input_channels: int,
    device: torch.device,
    precision: torch.dtype,
    seed: Optional[int],
    verbose: bool,
    **kwargs
) -> nn.Module:
    """Create pure CNN model (baseline)"""
    from src.models import CNN
    
    # Extract CNN-specific parameters
    feature_channels = kwargs.get('feature_channels', [64, 128, 256])
    dropout_rate = kwargs.get('dropout_rate', 0.3)
    
    model = CNN(
        input_channels=input_channels,
        feature_channels=feature_channels,
        output_dim=output_dim,
        device=device,
        precision=precision,
        seed=seed,
        dropout_rate=dropout_rate,
    )
    
    if verbose:
        print(f"  Feature channels: {feature_channels}")
        print(f"  Dropout rate: {dropout_rate}")
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model architecture.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    
    Returns
    -------
    dict
        Dictionary with model information and parameters
    """
    model_name = model_name.lower().replace('-', '').replace('_', '')
    
    info = {
        'spdnet': {
            'name': 'SPDNet',
            'description': 'Pure SPDNet for SPD matrix classification',
            'backbone': 'None (SPD layers only)',
            'best_for': 'Covariance matrix inputs',
            'parameters': {
                'input_dim': '(int) - Input SPD matrix dimension',
                'hidden_layers_size': '(List[int]) - Hidden layer dimensions',
                'eps': '(float) - Regularization for ReEig',
                'batchnorm': '(bool) - Use SPD batchnorm',
                'use_autograd': '(bool) - Use autograd for gradients',
                'dropout_rate': '(float) - Dropout rate before final layer',
            }
        },
        'cnnspdnet': {
            'name': 'CNN-SPDNet',
            'description': 'Custom CNN backbone + SPDNet for feature extraction',
            'backbone': 'Custom CNN (3 conv blocks)',
            'best_for': 'SAR images, small datasets',
            'parameters': {
                'feature_channels': '(List[int]) - CNN channel dimensions',
                'hidden_layers_size': '(List[int]) - SPDNet hidden dimensions',
                'shrinkage': "('none', 'ledoit_wolf') - Covariance shrinkage",
                'lbda': "('optimal' or float) - Shrinkage parameter",
                'eps': '(float) - Regularization value',
            }
        },
        'resnetspdnet': {
            'name': 'ResNet-SPDNet',
            'description': 'ResNet backbone + SPDNet',
            'backbone': 'ResNet (18/34/50/101/152)',
            'best_for': 'Large datasets, transfer learning',
            'parameters': {
                'resnet_type': "('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')",
                'num_layers': '(int) - Number of ResNet blocks (1-4)',
                'pretrained': '(bool) - Use ImageNet pretrained weights',
                'hidden_layers_size': '(List[int]) - SPDNet hidden dimensions',
                'shrinkage': "('ledoit_wolf', 'none') - Covariance shrinkage",
            }
        },
        'mobilenetspdnet': {
            'name': 'MobileNetV3-Large-SPDNet',
            'description': 'MobileNetV3 Large backbone + SPDNet',
            'backbone': 'MobileNetV3 Large',
            'best_for': 'Mobile deployment, efficiency',
            'parameters': {
                'num_layers': '(int) - Number of MobileNet blocks',
                'pretrained': '(bool) - Use ImageNet pretrained weights',
                'hidden_layers_size': '(List[int]) - SPDNet hidden dimensions',
                'shrinkage': "('ledoit_wolf', 'none') - Covariance shrinkage",
            }
        },
        'efficientnetb0spdnet': {
            'name': 'EfficientNet-B0-SPDNet',
            'description': 'EfficientNet-B0 backbone + SPDNet',
            'backbone': 'EfficientNet-B0',
            'best_for': 'Balanced accuracy and efficiency',
            'parameters': {
                'num_layers': '(int) - Number of EfficientNet blocks',
                'pretrained': '(bool) - Use ImageNet pretrained weights',
                'hidden_layers_size': '(List[int]) - SPDNet hidden dimensions',
                'shrinkage': "('ledoit_wolf', 'none') - Covariance shrinkage",
            }
        },
        'efficientnetb4spdnet': {
            'name': 'EfficientNet-B4-SPDNet',
            'description': 'EfficientNet-B4 backbone + SPDNet',
            'backbone': 'EfficientNet-B4',
            'best_for': 'High accuracy, larger models',
            'parameters': {
                'num_layers': '(int) - Number of EfficientNet blocks',
                'pretrained': '(bool) - Use ImageNet pretrained weights',
                'hidden_layers_size': '(List[int]) - SPDNet hidden dimensions',
                'shrinkage': "('ledoit_wolf', 'none') - Covariance shrinkage",
            }
        },
        'cnn': {
            'name': 'CNN',
            'description': 'Pure CNN without SPDNet (baseline)',
            'backbone': 'Custom CNN',
            'best_for': 'Baseline comparison',
            'parameters': {
                'feature_channels': '(List[int]) - CNN channel dimensions',
                'dropout_rate': '(float) - Dropout rate',
            }
        }
    }
    
    if model_name not in info:
        return {'error': f'Unknown model: {model_name}'}
    
    return info[model_name]


def list_available_models() -> List[str]:
    """
    List all available model architectures.
    
    Returns
    -------
    list
        List of available model names
    """
    return [
        'spdnet',
        'cnnspdnet',
        'resnetspdnet',
        'mobilenetspdnet',
        'efficientnetb0spdnet',
        'efficientnetb4spdnet',
        'cnn'
    ]


if __name__ == "__main__":
    print("="*60)
    print("Meta Models - Available Architectures")
    print("="*60)
    
    # List all models
    print("\nAvailable Models:")
    for model_name in list_available_models():
        info = get_model_info(model_name)
        print(f"\n {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Backbone: {info['backbone']}")
        print(f"   Best for: {info['best_for']}")
        print(f"   Parameters:")
        for param, desc in info['parameters'].items():
            print(f"      • {param}: {desc}")
    
    # Example usage
    print("\n" + "="*60)
    print("Example Usage")
    print("="*60)
    
    # Example 1: CNNSPDNet
    print("\n1. Creating CNNSPDNet for SAR classification:")
    try:
        model = create_model(
            'cnnspdnet',
            output_dim=10,
            input_channels=1,
            feature_channels=[64, 192],
            hidden_layers_size=[24],
            verbose=True
        )
        print(f"   ✓ Model created successfully!")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 2: ResNetSPDNet
    print("\n2. Creating ResNet18-SPDNet with pretrained backbone:")
    try:
        model = create_model(
            'resnetspdnet',
            output_dim=7,
            input_channels=3,
            resnet_type='resnet18',
            num_layers=2,
            pretrained=True,
            verbose=True
        )
        print(f"   ✓ Model created successfully!")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 3: Pure SPDNet
    print("\n3. Creating pure SPDNet for covariance matrices:")
    try:
        model = create_model(
            'spdnet',
            output_dim=4,
            input_dim=50,
            hidden_layers_size=[30, 20],
            batchnorm=True,
            verbose=True
        )
        print(f"   ✓ Model created successfully!")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Meta Models Ready!")
    print("="*60)

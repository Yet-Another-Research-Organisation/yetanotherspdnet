#!/usr/bin/env python3
"""
General DataLoader Interface for Multiple SAR/Hyperspectral Datasets
====================================================================

Unified interface for loading different datasets:
- FUSAR-Ship: Ship classification from SAR images
- TenGeoP-SARwv: 10 classes of oceanic/atmospheric phenomena
- S1SLC: Sentinel-1 SLC complex data (7 land cover classes)
- HyperLeaf: Hyperspectral crop classification/regression

Author: AI Assistant
Date: 2025-10-19
"""

import os
from typing import Tuple, Optional, Dict, Any, List
from torch.utils.data import DataLoader
from pathlib import Path


def general_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    val_ratio: float = 0.2,
    verbose: bool = False,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    General dataloader function that creates train, validation, and test dataloaders
    for various SAR and hyperspectral datasets.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Options: 'fusar', 'tengeop', 's1slc', 'hyperleaf', 'rices90', 'hdm05'
    data_root : str
        Path to the dataset root directory
    batch_size : int, default=32
        Batch size for all dataloaders
    num_workers : int, default=4
        Number of worker processes for data loading
    shuffle : bool, default=True
        Whether to shuffle training data
    seed : int, default=42
        Random seed for reproducibility
    val_ratio : float, default=0.2
        Ratio of training data to use for validation (0.0 to 1.0)
    verbose : bool, default=False
        Whether to print dataset information
    **dataset_kwargs : dict
        Additional dataset-specific arguments
        
    Dataset-specific parameters:
    
    FUSAR-Ship (dataset_name='fusar'):
        - class_level: str, default='basic'
            'basic' for main categories or 'fine' for sub-categories
        - max_samples: int, optional
            Maximum samples per class
        - min_samples: int, optional
            Minimum samples required per class (drop classes below this threshold)
        - resize_mode: str, default='min_size'
            'min_size' or 'fixed_size'
        - target_size: tuple, optional
            (height, width) for fixed_size mode
        - channels: str, default='single'
            'single' or 'rgb'
            
    TenGeoP-SARwv (dataset_name='tengeop'):
        - max_samples: int, optional
            Maximum samples per class
        - resize_mode: str, default='min_size'
            'min_size' or 'fixed_size'
        - target_size: tuple, optional
            (height, width) for fixed_size mode
        - channels: str, default='single'
            'single' or 'rgb'
            
    S1SLC (dataset_name='s1slc'):
        - mode: str, default='standard'
            'standard' for random split or 'domain_adaptation' for city-based split
        - max_samples_per_class: int, optional
            Maximum samples per class
        - train_cities: list, optional
            Cities for training (domain adaptation mode only)
        - test_city: str, optional
            City for testing (domain adaptation mode only)
        - transform: callable, optional
            Transformation to apply to samples
            
    HyperLeaf (dataset_name='hyperleaf'):
        - task_type: str, default='classification'
            'classification' or 'regression'
        - target_columns: list, optional
            Specific target columns to use
        - train_transform: callable, optional
            Transformations for training data
        - test_transform: callable, optional
            Transformations for test data
        - max_samples: int, optional
            Maximum samples per class for training
            
    Rices90 (dataset_name='rices90'):
        - classes: list, optional
            Specific classes to use
        - max_samples_per_class: int, optional
            Maximum samples per class
        - max_classes: int, optional
            Maximum number of classes to use
        - label_noise_ratio: float, default=0.0
            Ratio of labels to corrupt (0.0 to 1.0)
        - test_ratio: float, default=0.1
            Ratio of data for test set
    
    HDM05 (dataset_name='hdm05'):
        - max_samples_per_class: int, optional
            Maximum samples per class
        - max_classes: int, optional
            Maximum number of classes to use (out of 117)
        - label_noise_ratio: float, default=0.0
            Ratio of labels to corrupt (0.0 to 1.0)
        - test_ratio: float, default=0.15
            Ratio of data for test set
        - train_ratio: float, optional
            Ratio of data for training (auto-calculated if not provided)
    
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    test_loader : DataLoader
        Test dataloader
        
    Examples
    --------
    >>> # FUSAR-Ship basic classification
    >>> train, val, test = general_dataloader(
    ...     'fusar',
    ...     '~/data/FUSAR_Ship1.0',
    ...     batch_size=16,
    ...     class_level='basic',
    ...     max_samples=500
    ... )
    
    >>> # TenGeoP with custom size
    >>> train, val, test = general_dataloader(
    ...     'tengeop',
    ...     '~/data/TenGeoP-SARwv',
    ...     batch_size=32,
    ...     target_size=(256, 256),
    ...     resize_mode='fixed_size'
    ... )
    
    >>> # S1SLC domain adaptation
    >>> train, val, test = general_dataloader(
    ...     's1slc',
    ...     '~/data/s1slc.zarr',
    ...     mode='domain_adaptation',
    ...     test_city='Paris'
    ... )
    
    >>> # HyperLeaf regression
    >>> train, val, test = general_dataloader(
    ...     'hyperleaf',
    ...     '~/data/HyperLeaf2024',
    ...     task_type='regression'
    ... )
    """
    dataset_name = dataset_name.lower()
    data_root = os.path.expanduser(data_root)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        print(f"Data root: {data_root}")
        print(f"Batch size: {batch_size}")
        print(f"Validation ratio: {val_ratio}")
        print(f"Random seed: {seed}")
    
    # Route to appropriate dataloader
    if dataset_name in ['fusar', 'fusar-ship', 'fusarship']:
        return _load_fusar(
            data_root, batch_size, num_workers, shuffle, 
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    elif dataset_name in ['tengeop', 'tengeop-sarwv', 'tengeopsarwv']:
        return _load_tengeop(
            data_root, batch_size, num_workers, shuffle,
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    elif dataset_name in ['s1slc', 's1-slc', 'sentinel1']:
        return _load_s1slc(
            data_root, batch_size, num_workers, shuffle,
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    elif dataset_name in ['hyperleaf', 'hyper-leaf']:
        return _load_hyperleaf(
            data_root, batch_size, num_workers, shuffle,
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    elif dataset_name in ['rices90', 'rices-90', 'rice90', 'rice']:
        return _load_rices90(
            data_root, batch_size, num_workers, shuffle,
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    elif dataset_name in ['hdm05', 'hdm-05', 'hdm']:
        return _load_hdm05(
            data_root, batch_size, num_workers, shuffle,
            seed, val_ratio, verbose, **dataset_kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: 'fusar', 'tengeop', 's1slc', 'hyperleaf', 'rices90', 'hdm05'"
        )


def _load_fusar(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load FUSAR-Ship dataset"""
    from src.dataloader.fusar_dataloader import create_fusar_dataloader
    
    # Extract FUSAR-specific parameters
    class_level = kwargs.get('class_level', 'basic')
    max_samples = kwargs.get('max_samples', None)
    min_samples = kwargs.get('min_samples', None)
    resize_mode = kwargs.get('resize_mode', 'min_size')
    target_size = kwargs.get('target_size', None)
    channels = kwargs.get('channels', 'single')
    transform = kwargs.get('transform', None)
    load_metadata = kwargs.get('load_metadata', True)
    
    # Create train and test loaders (includes internal split)
    train_loader, val_loader, test_loader = create_fusar_dataloader(
        data_root=data_root,
        class_level=class_level,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        max_samples=max_samples,
        min_samples=min_samples,
        resize_mode=resize_mode,
        target_size=target_size,
        split_ratio=1.0 - val_ratio,  # Will be further split
        seed=seed,
        load_metadata=load_metadata,
        channels=channels,
        verbose=verbose,
        transform=transform
    )
    
    if verbose:
        print(f"✅ FUSAR-Ship loaded:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _load_tengeop(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load TenGeoP-SARwv dataset"""
    from src.dataloader.tengeop_dataloader import create_dataloader
    
    # Extract TenGeoP-specific parameters
    max_samples = kwargs.get('max_samples', None)
    resize_mode = kwargs.get('resize_mode', 'min_size')
    target_size = kwargs.get('target_size', None)
    channels = kwargs.get('channels', 'single')
    transform = kwargs.get('transform', None)
    load_metadata = kwargs.get('load_metadata', True)
    split_ratio = kwargs.get('split_ratio', 0.8)
    
    # Create dataloaders (now returns train, val, test)
    train_loader, val_loader, test_loader = create_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        max_samples=max_samples,
        resize_mode=resize_mode,
        target_size=target_size,
        split_ratio=split_ratio,
        val_ratio=val_ratio,
        seed=seed,
        load_metadata=load_metadata,
        channels=channels,
        verbose=verbose,
        transform=transform
    )
    
    if verbose:
        print(f"✅ TenGeoP-SARwv loaded:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _load_s1slc(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load S1SLC dataset"""
    from src.dataloader.s1slc_zarr_dataloader import (
        create_s1slc_dataloader,
        create_s1slc_DA_dataloader
    )
    import torch
    from torch.utils.data import random_split
    
    mode = kwargs.get('mode', 'standard')
    max_samples_per_class = kwargs.get('max_samples_per_class', None)
    transform = kwargs.get('transform', None)
    train_ratio = kwargs.get('train_ratio', 0.8)
    
    if mode == 'domain_adaptation':
        # Domain adaptation mode: train on multiple cities, test on one
        train_cities = kwargs.get('train_cities', None)
        test_city = kwargs.get('test_city', None)
        
        train_loader, test_loader = create_s1slc_DA_dataloader(
            zarr_path=data_root,
            train_cities=train_cities,
            test_city=test_city,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            shuffle_train=shuffle,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            verbose=verbose
        )
    else:
        # Standard mode: random split
        train_loader, test_loader = create_s1slc_dataloader(
            zarr_path=data_root,
            batch_size=batch_size,
            train_ratio=train_ratio,
            seed=seed,
            num_workers=num_workers,
            shuffle_train=shuffle,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            verbose=verbose
        )
    
    # Split train into train/val
    train_dataset = train_loader.dataset
    train_size = int(len(train_dataset) * (1 - val_ratio))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    if verbose:
        print(f"✅ S1SLC loaded (mode: {mode}):")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _load_hyperleaf(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load HyperLeaf dataset"""
    from src.dataloader.hyperleaf_dataloader import create_hyperleaf_dataloaders
    import torch
    from torch.utils.data import random_split
    
    task_type = kwargs.get('task_type', 'classification')
    target_columns = kwargs.get('target_columns', None)
    train_transform = kwargs.get('train_transform', None)
    test_transform = kwargs.get('test_transform', None)
    max_samples = kwargs.get('max_samples', None)
    test_ratio = kwargs.get('test_ratio', 0.2)
    images_dir = kwargs.get('images_dir', 'images')
    label_noise_ratio = kwargs.get('label_noise_ratio', 0.0)

    if isinstance(train_transform, str):
        if train_transform.lower() == 'none':
            train_transform = None
    if isinstance(test_transform, str):
        if test_transform.lower() == 'none':
            test_transform = None
    # Create train and test loaders
    train_loader, val_loader, test_loader = create_hyperleaf_dataloaders(
        data_dir=data_root,
        task_type=task_type,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        test_transform=test_transform,
        target_columns=target_columns,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        max_samples=max_samples,
        images_dir=images_dir,
        label_noise_ratio=label_noise_ratio
    )
    
    if verbose:
        print(f"✅ HyperLeaf loaded (task: {task_type}):")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _load_rices90(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Rices90 dataset"""
    from src.dataloader.rices90_dataloader import create_rices90_dataloaders
    
    classes = kwargs.get('classes', None)
    max_samples_per_class = kwargs.get('max_samples_per_class', None)
    max_classes = kwargs.get('max_classes', None)
    label_noise_ratio = kwargs.get('label_noise_ratio', 0.0)
    test_ratio = kwargs.get('test_ratio', 0.1)
    
    # Create train/val/test loaders
    train_loader, val_loader, test_loader = create_rices90_dataloaders(
        data_dir=data_root,
        classes=classes,
        max_samples_per_class=max_samples_per_class,
        max_classes=max_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        label_noise_ratio=label_noise_ratio
    )
    
    if verbose:
        print(f"✅ Rices90 loaded:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _load_hdm05(
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    val_ratio: float,
    verbose: bool,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load HDM05 dataset"""
    from src.dataloader.hdm05_dataloader import create_hdm05_dataloaders
    
    max_samples_per_class = kwargs.get('max_samples_per_class', None)
    max_classes = kwargs.get('max_classes', None)
    label_noise_ratio = kwargs.get('label_noise_ratio', 0.0)
    test_ratio = kwargs.get('test_ratio', 0.15)
    train_ratio = kwargs.get('train_ratio', None)
    
    # Calculate train ratio if not provided
    if train_ratio is None:
        train_ratio = 1.0 - val_ratio - test_ratio
    
    # Create train/val/test loaders
    train_loader, val_loader, test_loader = create_hdm05_dataloaders(
        data_dir=data_root,
        max_samples_per_class=max_samples_per_class,
        max_classes=max_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        label_noise_ratio=label_noise_ratio,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"✅ HDM05 loaded:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about available parameters for a dataset
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    
    Returns
    -------
    dict
        Dictionary with dataset information and parameters
    """
    info = {
        'fusar': {
            'name': 'FUSAR-Ship',
            'description': 'Ship classification from SAR images',
            'classes': 'Variable (basic: 5+, fine: 15+)',
            'parameters': {
                'class_level': "('basic', 'fine') - Classification granularity",
                'max_samples': '(int) - Max samples per class',
                'min_samples': '(int) - Min samples required per class (drop below)',
                'resize_mode': "('min_size', 'fixed_size') - Resize strategy",
                'target_size': '(tuple) - Target image size',
                'channels': "('single', 'rgb') - Number of channels",
            }
        },
        'tengeop': {
            'name': 'TenGeoP-SARwv',
            'description': '10 classes of oceanic/atmospheric phenomena',
            'classes': 10,
            'parameters': {
                'max_samples': '(int) - Max samples per class',
                'resize_mode': "('min_size', 'fixed_size') - Resize strategy",
                'target_size': '(tuple) - Target image size',
                'channels': "('single', 'rgb') - Number of channels",
            }
        },
        's1slc': {
            'name': 'Sentinel-1 SLC',
            'description': '7 land cover classes from complex SAR data',
            'classes': 7,
            'parameters': {
                'mode': "('standard', 'domain_adaptation') - Split mode",
                'max_samples_per_class': '(int) - Max samples per class',
                'train_cities': '(list) - Cities for training (DA mode)',
                'test_city': '(str) - City for testing (DA mode)',
                'transform': '(callable) - Data transformation',
            }
        },
        'hyperleaf': {
            'name': 'HyperLeaf',
            'description': 'Hyperspectral crop classification/regression',
            'classes': '4 (classification) or 4 targets (regression)',
            'parameters': {
                'task_type': "('classification', 'regression') - Task type",
                'target_columns': '(list) - Specific target columns',
                'train_transform': '(callable) - Training transformation',
                'test_transform': '(callable) - Test transformation',
                'max_samples': '(int) - Max samples per class for training',
            }
        },
        'rices90': {
            'name': 'Rices90',
            'description': '90 Vietnamese rice varieties with covariance matrices 256x256',
            'classes': 90,
            'parameters': {
                'classes': '(list) - Specific classes to use',
                'max_samples_per_class': '(int) - Max samples per class',
                'max_classes': '(int) - Max number of classes to use',
                'label_noise_ratio': '(float) - Ratio of labels to corrupt (0.0-1.0)',
                'test_ratio': '(float) - Ratio of data for test set',
            }
        },
        'hdm05': {
            'name': 'HDM05',
            'description': '117 human motion capture classes with SPD matrices 93x93',
            'classes': 117,
            'parameters': {
                'max_samples_per_class': '(int) - Max samples per class',
                'max_classes': '(int) - Max number of classes to use',
                'label_noise_ratio': '(float) - Ratio of labels to corrupt (0.0-1.0)',
                'test_ratio': '(float) - Ratio of data for test set',
                'train_ratio': '(float) - Ratio of data for training (auto-calculated if not provided)',
            }
        }
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in info:
        return {'error': f'Unknown dataset: {dataset_name}'}
    
    return info[dataset_name]


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("General DataLoader Examples")
    print("="*60)
    
    # Print available datasets
    print("\nAvailable datasets:")
    for name in ['fusar', 'tengeop', 's1slc', 'hyperleaf', 'rices90', 'hdm05']:
        info = get_dataset_info(name)
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Classes: {info['classes']}")
        print(f"  Parameters:")
        for param, desc in info['parameters'].items():
            print(f"    - {param}: {desc}")
    
    # Example 1: FUSAR-Ship
    print("\n" + "="*60)
    print("Example 1: FUSAR-Ship")
    print("="*60)
    try:
        train, val, test = general_dataloader(
            'fusar',
            '~/Documents/Dataset/FUSAR_Ship1.0',
            batch_size=16,
            class_level='basic',
            max_samples=100,
            verbose=True
        )
        print(f"✓ Successfully created dataloaders")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: TenGeoP
    print("\n" + "="*60)
    print("Example 2: TenGeoP-SARwv")
    print("="*60)
    try:
        train, val, test = general_dataloader(
            'tengeop',
            '~/Documents/Dataset/TenGeoP-SARwv',
            batch_size=32,
            max_samples=50,
            verbose=True
        )
        print(f"✓ Successfully created dataloaders")
    except Exception as e:
        print(f"✗ Error: {e}")

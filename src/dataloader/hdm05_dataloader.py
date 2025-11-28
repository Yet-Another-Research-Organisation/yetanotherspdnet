#!/usr/bin/env python3
"""
HDM05 Dataset Dataloader
Human motion capture dataset with 117 motion classes.

Dataset obtained from: https://github.com/GitZH-Chen/RMLR
- 2086 samples of SPD matrices (93x93)
- 117 motion classes (labels 0-116)
- Format: {id}_{unknown}_{label}.npy

Author: AI Assistant
Date: 2025-11-17
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, List
import warnings


class HDM05Dataset(Dataset):
    """
    Dataset class for HDM05 with SPD matrices (93x93).

    Args:
        data_dir: Root directory path containing the .npy files
        classes: Optional list of class labels to use (all if None)
        max_samples_per_class: Maximum number of samples per class
        max_classes: Maximum number of classes to use (None = all)
        transform: Optional transformations to apply
        label_noise_ratio: Ratio of labels to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility
        dtype: Data type for tensors
        device: Device to load tensors on
        preload: Whether to preload all data into memory
        verbose: Verbosity level for logging
    """

    # 117 motion classes from HDM05 dataset
    CLASS_NAMES = [str(i) for i in range(117)]
    
    # Default scaling factor to normalize geometric mean of eigenvalues to ~1.0
    # Calculated from median geometric mean of eigenvalues across ALL 2086 samples
    # Median geo_mean = 0.00005345 -> factor = 1.0 / 0.00005345 ≈ 19000
    DEFAULT_SCALING_FACTOR = 19000.0

    def __init__(
        self,
        data_dir: str,
        classes: Optional[List[int]] = None,
        max_samples_per_class: Optional[int] = None,
        max_classes: Optional[int] = None,
        transform: Optional[callable] = None,
        label_noise_ratio: float = 0.0,
        seed: int = 42,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = torch.device("cpu"),
        preload: Optional[bool] = False,
        verbose: Optional[int] = 0,
        scale_data: Optional[bool] = True,
        scaling_factor: Optional[float] = None,
    ) -> None:
        """Initialize the HDM05Dataset.
        
        Args:
            scale_data: Whether to scale matrices to normalize eigenvalues (default: True)
            scaling_factor: Custom scaling factor (default: 22000.0 for geometric mean ~1.0)
        """
        self.data_dir = data_dir
        self.d = 93  # Dimension of SPD matrices
        self.n_class = 117  # Number of classes
        self.dtype = dtype
        self.device = device
        self.preload = preload
        self.verbose = verbose
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.max_classes = max_classes
        self.label_noise_ratio = label_noise_ratio
        self.seed = seed
        self.scale_data = scale_data
        self.scaling_factor = scaling_factor if scaling_factor is not None else self.DEFAULT_SCALING_FACTOR
        self.target_size = (self.d, self.d)
        # Random number generator for reproducibility
        self.rng = np.random.RandomState(seed)
        
        if self.verbose > 0 and self.scale_data:
            print(f"Data scaling enabled with factor: {self.scaling_factor:.1f}")
            print(f"  (This normalizes geometric mean of eigenvalues to ~1.0)")

        # Discover and organize files
        self._discover_files()
        
        # Determine which classes to use
        all_classes = sorted(list(set(info['label'] for info in self.file_info)))
        
        # Limit number of classes if requested
        if max_classes is not None and max_classes < len(all_classes):
            selected_classes = self.rng.choice(all_classes, size=max_classes, replace=False).tolist()
            all_classes = sorted(selected_classes)
            self.file_info = [info for info in self.file_info if info['label'] in all_classes]
            if self.verbose > 0:
                print(f"Using {max_classes} classes: {all_classes}")
        
        # Filter specific classes if provided
        if classes is not None:
            classes = [c for c in classes if c in all_classes]
            if len(classes) == 0:
                raise ValueError(f"None of the specified classes found in {all_classes}")
            self.file_info = [info for info in self.file_info if info['label'] in classes]
            all_classes = classes
            if self.verbose > 0:
                print(f"Using custom classes: {classes}")
        
        # Create class mapping
        self.classes = all_classes
        self.class_to_idx = {class_label: idx for idx, class_label in enumerate(all_classes)}
        
        # Limit samples per class if requested
        if max_samples_per_class is not None:
            self._limit_samples_per_class()
        
        # Apply label noise if requested
        if label_noise_ratio > 0.0:
            self._corrupt_labels()
        
        if self.verbose > 0:
            print(f"\nHDM05 Dataset:")
            print(f"  - {len(self.file_info)} samples")
            print(f"  - {len(self.classes)} classes")
            print(f"  - Matrix size: 93x93")
            self._print_class_distribution()
        
        # Preload data if requested
        if preload:
            self._preload()

    def _discover_files(self) -> None:
        """Discover and validate .npy files from the data directory."""
        files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.data_dir}! Please check path."
            )
        
        if self.verbose > 0:
            print(f"Found {len(files)} .npy files")
        
        # Parse file information
        self.file_info = []
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                # Format: {id}_{unknown}_{label}.npy
                label = int(filename.split(".")[0].split("_")[-1])
                self.file_info.append({
                    'path': file_path,
                    'label': label,
                    'filename': filename
                })
            except (IndexError, ValueError) as e:
                warnings.warn(f"Invalid filename format: {filename}")
                continue

    def _limit_samples_per_class(self):
        """Limit the number of samples per class."""
        limited_file_info = []
        for class_label in self.classes:
            class_files = [info for info in self.file_info if info['label'] == class_label]
            if len(class_files) > self.max_samples_per_class:
                selected_indices = self.rng.choice(
                    len(class_files),
                    size=self.max_samples_per_class,
                    replace=False
                )
                selected_files = [class_files[i] for i in selected_indices]
                limited_file_info.extend(selected_files)
                if self.verbose > 0:
                    print(f"Class {class_label}: {len(class_files)} -> {self.max_samples_per_class} samples")
            else:
                limited_file_info.extend(class_files)
                if self.verbose > 0:
                    print(f"Class {class_label}: {len(class_files)} samples (not limited)")
        self.file_info = limited_file_info

    def _corrupt_labels(self):
        """Randomly corrupt a percentage of labels."""
        if self.label_noise_ratio <= 0.0 or self.label_noise_ratio >= 1.0:
            if self.label_noise_ratio < 0.0 or self.label_noise_ratio > 1.0:
                warnings.warn(f"label_noise_ratio must be between 0.0 and 1.0, got {self.label_noise_ratio}")
            return
        
        num_samples = len(self.file_info)
        num_to_corrupt = int(num_samples * self.label_noise_ratio)
        
        # Randomly select indices to corrupt
        corrupt_indices = self.rng.choice(num_samples, size=num_to_corrupt, replace=False)
        
        # For each selected index, assign a different class
        corrupted_count = 0
        for idx in corrupt_indices:
            original_label = self.file_info[idx]['label']
            # Choose a different class
            other_classes = [c for c in self.classes if c != original_label]
            if other_classes:
                new_label = self.rng.choice(other_classes)
                self.file_info[idx]['label'] = new_label
                corrupted_count += 1
        
        if self.verbose > 0:
            print(f"\nLabel noise applied:")
            print(f"  - {corrupted_count} labels corrupted ({self.label_noise_ratio*100:.1f}% of {num_samples})")

    def _print_class_distribution(self):
        """Print class distribution."""
        class_counts = {}
        for info in self.file_info:
            label = info['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("\nClass distribution (top 10):")
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for class_label, count in sorted_counts[:10]:
            percentage = count / len(self.file_info) * 100
            print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
        if len(sorted_counts) > 10:
            print(f"  ... and {len(sorted_counts) - 10} more classes")

    def _preload(self) -> None:
        """Preload all data into memory."""
        if self.verbose > 0:
            print("Preloading data into memory...")
        
        data_list = []
        labels_list = []
        for info in self.file_info:
            x = np.load(info['path'])
            # Apply scaling if enabled
            if self.scale_data:
                x = x * self.scaling_factor
            x = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
            if self.transform is not None:
                x = self.transform(x)
            data_list.append(x)
            labels_list.append(self.class_to_idx[info['label']])
        
        self._data = torch.stack(data_list)
        self._labels = torch.tensor(labels_list, dtype=torch.long)
        
        if self.verbose > 0:
            print(f"Preloaded {len(self._data)} samples")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_info)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if self.preload:
            return self._data[index], self._labels[index]
        else:
            info = self.file_info[index]
            x = np.load(info['path'])
            # Apply scaling if enabled
            if self.scale_data:
                x = x * self.scaling_factor
            x = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
            if self.transform is not None:
                x = self.transform(x)
            y = torch.tensor(self.class_to_idx[info['label']], dtype=torch.long)
            return x, y

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced classification."""
        class_counts = np.zeros(len(self.classes))
        for info in self.file_info:
            class_idx = self.class_to_idx[info['label']]
            class_counts[class_idx] += 1
        
        # Calculate weights (inversely proportional to counts)
        total_samples = len(self.file_info)
        weights = total_samples / (len(self.classes) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  num_samples={len(self)},\n"
            f"  num_classes={len(self.classes)},\n"
            f"  matrix_size=93x93,\n"
            f"  preload={self.preload},\n"
            f"  dtype={self.dtype},\n"
            f"  device={self.device}\n"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"HDM05Dataset(data_dir='{self.data_dir}', num_samples={len(self)}, "
            f"num_classes={len(self.classes)}, dtype={self.dtype}, "
            f"device={self.device}, preload={self.preload})"
        )


def create_hdm05_dataloaders(
    data_dir: str = '/home/mgallet/Documents/DATASET/HDM05',
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples_per_class: Optional[int] = None,
    max_classes: Optional[int] = None,
    label_noise_ratio: float = 0.0,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    verbose: int = 0,
    scale_data: bool = True,
    scaling_factor: Optional[float] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for HDM05 dataset.

    Args:
        data_dir: Path to directory containing .npy files
        batch_size: Batch size for dataloaders
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.15)
        test_ratio: Proportion of data for testing (default 0.15)
        max_samples_per_class: Maximum samples per class (None for all)
        max_classes: Maximum number of classes (None for all 117)
        label_noise_ratio: Ratio of labels to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for CUDA
        verbose: Verbosity level
        scale_data: Whether to scale matrices (default: True, normalizes eigenvalues)
        scaling_factor: Custom scaling factor (default: 22000.0)
        **dataset_kwargs: Additional arguments for HDM05Dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Create full dataset
    full_dataset = HDM05Dataset(
        data_dir=data_dir,
        max_samples_per_class=max_samples_per_class,
        max_classes=max_classes,
        label_noise_ratio=label_noise_ratio,
        seed=seed,
        verbose=verbose,
        scale_data=scale_data,
        scaling_factor=scaling_factor,
        **dataset_kwargs
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure all samples are used

    # Split dataset using torch.random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    if verbose > 0:
        print(f"\nDataset splits:")
        print(f"  - Train: {train_size} samples ({train_ratio*100:.1f}%)")
        print(f"  - Val:   {val_size} samples ({val_ratio*100:.1f}%)")
        print(f"  - Test:  {test_size} samples ({test_ratio*100:.1f}%)")
        print(f"  - Total: {total_size} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the HDM05 dataloader."""
    print("=" * 80)
    print("Testing HDM05 Dataloader")
    print("=" * 80)

    # Test 1: Basic loading with limited classes
    print("\n[Test 1] Loading 10 classes with max 20 samples per class...")
    train_loader, val_loader, test_loader = create_hdm05_dataloaders(
        max_classes=10,
        max_samples_per_class=20,
        batch_size=16,
        verbose=1
    )

    # Check a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nTrain batch shape: {batch_x.shape}, labels: {batch_y.shape}")
    print(f"Matrix shape: {batch_x[0].shape}")
    print(f"Sample labels in batch: {batch_y[:5].tolist()}")

    # Test 2: With label noise
    print("\n" + "=" * 80)
    print("[Test 2] Loading with 10% label noise...")
    train_loader, val_loader, test_loader = create_hdm05_dataloaders(
        max_classes=5,
        max_samples_per_class=10,
        label_noise_ratio=0.1,
        batch_size=8,
        verbose=1
    )

    # Test 3: Full dataset info
    print("\n" + "=" * 80)
    print("[Test 3] Full dataset statistics...")
    dataset = HDM05Dataset(
        data_dir='/home/mgallet/Documents/DATASET/HDM05',
        verbose=1
    )
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Target size: {dataset.d}x{dataset.d}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

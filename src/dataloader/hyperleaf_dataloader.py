"""
HyperLeaf Dataset Dataloader
Supports both classification and regression tasks.

Classification: One-hot encoding of columns [Heerup, Kvium, Rembrandt, Sheriff]
Regression: Prediction of [GrainWeight, Gsw, PhiPS2, Fertilizer]
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Union, List
from sklearn.covariance import ledoit_wolf
import warnings
import tifffile
from torch import nn



class ProcessingCov_HyperLeaf2024(nn.Module):
    def __init__(self, exclude_zero: bool = True):
        super().__init__()
        self.exclude_zero = exclude_zero

    def __call__(self, X: np.ndarray) -> np.ndarray:
        is_tensor = isinstance(X, torch.Tensor)
        if is_tensor:
            X = X.numpy()
        
        if X.ndim == 3:
            images = X.reshape(-1, X.shape[-1])  # Reshape en 2D (pixels, bands)
        elif X.ndim == 4:
            images = X.reshape(X.shape[0], -1, X.shape[-1])
        if self.exclude_zero:

            if X.ndim == 3:
                data = images[~np.all(images == 0, axis=1)]

            if X.ndim == 4:
                data = ([images[i, ~np.all(images[i] == 0, axis=1)] for i in range(images.shape[0])])
        
        out = ledoit_wolf(data)[0] if X.ndim == 3 else np.array([ledoit_wolf(d)[0] for d in data])
        if is_tensor:
            out = torch.tensor(out)
        return out


class ProcessingCov_HyperLeaf2024_SCM(nn.Module):
    """
    Version PyTorch pure qui calcule la matrice de covariance empirique (SCM) manuellement.
    Reste en torch sans conversion numpy pour de meilleures performances GPU.
    """
    def __init__(self, exclude_zero: bool = True):
        super().__init__()
        self.exclude_zero = exclude_zero

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calcule la matrice de covariance empirique (SCM).
        
        Args:
            X: Tensor de shape (H, W, C) pour une image ou (B, H, W, C) pour un batch
        
        Returns:
            Matrice de covariance de shape (C, C) ou (B, C, C)
        """
        original_device = X.device
        
        if X.ndim == 3:
            # Single image: (H, W, C) -> (N, C) où N = H*W (ou moins si exclusion de zéros)
            images = X.reshape(-1, X.shape[-1])  # (H*W, C)
            
            if self.exclude_zero:
                # Exclure les pixels où toutes les bandes sont à zéro
                mask = ~torch.all(images == 0, dim=1)  # (H*W,)
                data = images[mask]  # (N', C) où N' <= H*W
            else:
                data = images
            
            # Centrer les données
            data_centered = data - data.mean(dim=0, keepdim=True)  # (N', C)
            
            # Calculer la covariance: (1/N) * X^T @ X
            n_samples = data_centered.shape[0]
            cov = (data_centered.T @ data_centered) / n_samples  # (C, C)
            
            return cov
            
        elif X.ndim == 4:
            # Batch: (B, H, W, C) -> (B, N, C)
            batch_size = X.shape[0]
            images = X.reshape(batch_size, -1, X.shape[-1])  # (B, H*W, C)
            
            cov_matrices = []
            for i in range(batch_size):
                img = images[i]  # (H*W, C)
                
                if self.exclude_zero:
                    # Exclure les pixels où toutes les bandes sont à zéro
                    mask = ~torch.all(img == 0, dim=1)  # (H*W,)
                    data = img[mask]  # (N', C)
                else:
                    data = img
                
                # Centrer les données
                data_centered = data - data.mean(dim=0, keepdim=True)  # (N', C)
                
                # Calculer la covariance: (1/N) * X^T @ X
                n_samples = data_centered.shape[0]
                cov = (data_centered.T @ data_centered) / n_samples  # (C, C)
                
                cov_matrices.append(cov)
            
            return torch.stack(cov_matrices, dim=0)  # (B, C, C)
        
        else:
            raise ValueError(f"Expected input of shape (H, W, C) or (B, H, W, C), got {X.shape}")
    
class HyperLeafDataset(Dataset):
    """
    Dataset class for HyperLeaf data.
    
    Args:
        data_dir: Path to the HyperLeaf2024 folder
        split: 'train' or 'test'
        task_type: 'classification' or 'regression'
        transform: Optional image transforms
        target_columns: Optional list of target columns to use (auto-selected based on task_type if None)
        label_noise_ratio: Ratio of labels to corrupt randomly (0.0 to 1.0, only for train split)
    """
    CLASS_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 task_type: str = 'classification',
                 transform: Optional[transforms.Compose] = None,
                 target_columns: Optional[List[str]] = None,
                 max_samples: Optional[int] = None,
                 images_dir: str = 'images',
                 label_noise_ratio: float = 0.0):
        
        self.data_dir = data_dir
        self.split = split
        self.task_type = task_type
        self.transform = transform
        self.max_samples = max_samples
        self.type_data = images_dir
        self.images_dir = os.path.join(data_dir, images_dir)
        self.label_noise_ratio = label_noise_ratio
        self.target_size = (204, 204)  # Height, Width  Covariance matrices are 204x204
        self.class_to_idx = { name: idx for idx, name in enumerate(self.CLASS_NAMES) }
        
        # Validate inputs
        assert split in ['train', 'test'], f"Split must be 'train' or 'test', got {split}"
        assert task_type in ['classification', 'regression'], f"Task type must be 'classification' or 'regression', got {task_type}"
        assert os.path.exists(self.images_dir), f"Images directory not found: {self.images_dir}"
        
        # Load data
        if split == 'train':
            csv_path = os.path.join(data_dir, 'train.csv')
            self.data = pd.read_csv(csv_path, dtype={'ImageId': str})
            self.has_labels = True
        else:
            csv_path = os.path.join(data_dir, 'test.csv')
            self.data = pd.read_csv(csv_path, dtype={'ImageId': str})
            self.has_labels = False
        
        # Define target columns based on task type
        if target_columns is None:
            if task_type == 'classification':
                self.target_columns = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']
            else:  # regression
                self.target_columns = ['GrainWeight', 'Gsw', 'PhiPS2', 'Fertilizer']
        else:
            self.target_columns = target_columns
        
        # Validate target columns exist in training data
        if self.has_labels:
            missing_cols = [col for col in self.target_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Target columns not found in data: {missing_cols}")
        
        # Apply max_samples limit per class if specified (only for training split and classification)
        if self.has_labels and self.max_samples is not None and self.task_type == 'classification':
            self._limit_samples_per_class()
        
        # Apply label noise if specified (only for training split and classification)
        if self.has_labels and self.label_noise_ratio > 0.0 and self.split == 'train' and self.task_type == 'classification':
            self._corrupt_labels()
        
        # Set default transform if none provided
        if self.transform is None:
            # For hyperspectral data, we use different transforms
            self.transform = transforms.Compose([
                # Custom transform for hyperspectral data
                # Note: Standard transforms expect 3-channel RGB images
                # For hyperspectral data, we might need custom preprocessing
            ])
        
        print(f"Loaded {len(self.data)} samples for {split} split")
        if self.has_labels:
            print(f"Task type: {task_type}")
            print(f"Target columns: {self.target_columns}")
            if task_type == 'classification':
                self._print_class_distribution()
            else:
                self._print_regression_stats()
    
    def _limit_samples_per_class(self):
        """Limit the number of samples per class to max_samples."""
        # Convert one-hot to class labels
        class_labels = self.data[self.target_columns].values.argmax(axis=1)
        
        # Get indices for each class
        selected_indices = []
        for class_idx in range(len(self.target_columns)):
            class_mask = (class_labels == class_idx)
            class_indices = np.where(class_mask)[0]
            
            # Limit to max_samples per class
            if len(class_indices) > self.max_samples:
                # Randomly sample max_samples indices
                np.random.seed(42)  # For reproducibility
                selected_class_indices = np.random.choice(
                    class_indices, 
                    size=self.max_samples, 
                    replace=False
                )
            else:
                selected_class_indices = class_indices
            
            selected_indices.extend(selected_class_indices)
        
        # Sort indices to maintain original order
        selected_indices = sorted(selected_indices)
        
        # Filter the dataframe
        original_size = len(self.data)
        self.data = self.data.iloc[selected_indices].reset_index(drop=True)
        
        print(f"Limited dataset from {original_size} to {len(self.data)} samples ({self.max_samples} per class)")
    
    def _corrupt_labels(self):
        """
        Randomly corrupt a percentage of labels by assigning them to different classes.
        Only applied to training split for classification tasks.
        """
        if self.label_noise_ratio <= 0.0 or self.label_noise_ratio >= 1.0:
            if self.label_noise_ratio < 0.0 or self.label_noise_ratio > 1.0:
                raise ValueError(f"label_noise_ratio must be between 0.0 and 1.0, got {self.label_noise_ratio}")
            return
        
        num_classes = len(self.target_columns)
        num_samples = len(self.data)
        num_to_corrupt = int(num_samples * self.label_noise_ratio)
        
        # Get current labels (one-hot to class indices)
        current_labels = self.data[self.target_columns].values.argmax(axis=1)
        
        # Randomly select indices to corrupt
        np.random.seed(42)  # For reproducibility
        corrupt_indices = np.random.choice(num_samples, size=num_to_corrupt, replace=False)
        
        # For each selected index, assign a random different class
        corrupted_count = 0
        for idx in corrupt_indices:
            original_class = current_labels[idx]
            # Choose a different class randomly
            other_classes = [c for c in range(num_classes) if c != original_class]
            new_class = np.random.choice(other_classes)
            
            # Update the one-hot encoding
            self.data.loc[idx, self.target_columns] = 0
            self.data.loc[idx, self.target_columns[new_class]] = 1
            corrupted_count += 1
        
        print(f"Corrupted {corrupted_count} labels ({self.label_noise_ratio*100:.1f}% of {num_samples} samples)")
        print(f"Label noise applied: {num_to_corrupt} samples will have incorrect labels")
    
    def _print_class_distribution(self):
        """Print class distribution for classification task."""
        if not self.has_labels:
            return
        
        # Convert one-hot to class labels
        class_labels = self.data[self.target_columns].values.argmax(axis=1)
        class_names = self.target_columns
        
        print("Class distribution:")
        for i, class_name in enumerate(class_names):
            count = (class_labels == i).sum()
            percentage = count / len(class_labels) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def _print_regression_stats(self):
        """Print statistics for regression targets."""
        if not self.has_labels:
            return
        
        print("Regression target statistics:")
        for col in self.target_columns:
            values = self.data[col]
            print(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}, "
                  f"min={values.min():.3f}, max={values.max():.3f}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Get image ID
        image_id = self.data.iloc[idx]['ImageId']
        
        # Load image
        if self.type_data == 'cov':
            image_path = os.path.join(self.images_dir, f"{image_id}.pt")
        else:
            image_path = os.path.join(self.images_dir, f"{image_id}.tiff")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load hyperspectral image using tifffile for proper multi-page TIFF support
        if self.type_data == 'cov':
            image_array = torch.load(image_path).numpy()
        else:
            try:
                # Try using tifffile first (better for scientific data)
                image_array = tifffile.imread(image_path)
            except Exception:
                    with Image.open(image_path) as img:
                        frames = []
                        for i in range(img.n_frames):
                            img.seek(i)
                            frames.append(np.array(img))
                        image_array = np.stack(frames, axis=0)
            
        # Expected shape should be (204, 48, 352) based on your description
        # Debug: print(f"Loaded image shape: {image_array.shape}")
        
        # Convert to float32 and normalize
        image_array = image_array.astype(np.float32)
        
        # # Normalize to [0, 1] range
        # if image_array.max() > 1.0:
        #     image_array = image_array / 65535.0  # Assuming 16-bit images
        image_array = np.moveaxis(image_array, 0, -1)  # Change shape to (48, 352, 204)
        # Convert to tensor - shape will be (48, 352, 204)

        image = torch.tensor(image_array)
        
        # Apply additional transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        
        # Return image only for test set
        if not self.has_labels:
            return image
        
        # Get targets for training set
        targets = self.data.iloc[idx][self.target_columns].values.astype(np.float32)
        # evaluate one hot encode name is the same that class_to_idx

        targets = torch.tensor(targets)
       
        # For classification, ensure targets are one-hot encoded
        if self.task_type == 'classification':
            # Verify one-hot encoding (should sum to 1)
            if not np.allclose(targets.sum().item(), 1.0):
                warnings.warn(f"Sample {idx} has non-one-hot targets: {targets}")
            targets = targets.argmax().long()  # Single class index instead of one-hot
            
        
        return image, targets
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Calculate class weights for imbalanced classification.
        Returns None for regression tasks or test split.
        """
        if not self.has_labels or self.task_type != 'classification':
            return None
        
        # Convert one-hot to class labels
        class_labels = self.data[self.target_columns].values.argmax(axis=1)
        
        # Calculate class weights
        class_counts = np.bincount(class_labels)
        total_samples = len(class_labels)
        num_classes = len(self.target_columns)
        
        weights = total_samples / (num_classes * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_image_id(self, idx: int) -> str:
        """Get the image ID for a given index."""
        return self.data.iloc[idx]['ImageId']


def create_hyperleaf_dataloaders(data_dir: str,
                                task_type: str = 'classification',
                                batch_size: int = 32,
                                num_workers: int = 4,
                                train_transform = "CovTransform",
                                test_transform = "CovTransform",
                                target_columns: Optional[List[str]] = None,
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.2,
                                seed: int = 42,
                                use_only_train_csv= True,
                                max_samples: Optional[int] = None,
                                images_dir: str = 'images',
                                label_noise_ratio: float = 0.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for HyperLeaf dataset.
    
    Args:
        data_dir: Path to HyperLeaf2024 folder
        task_type: 'classification' or 'regression'
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_transform: Transform for training data (default: augmented transforms)
        test_transform: Transform for test data (default: basic transforms)
        target_columns: Optional list of target columns to use
        val_ratio: Ratio of training data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
        max_samples: Maximum number of samples per class for training (default: None = use all)
        label_noise_ratio: Ratio of training labels to corrupt (0.0 to 1.0, default: 0.0)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    from torch.utils.data import random_split
    
    # Default transforms for hyperspectral data
    if train_transform is None:
        train_transform = transforms.Compose([
            # For hyperspectral data, we mainly focus on normalization
            # Spatial augmentations can be added if needed
        ])
    elif train_transform == "CovTransform":
        train_transform = transforms.Compose([
            ProcessingCov_HyperLeaf2024(exclude_zero=True),
        ])
    elif train_transform == "CovTransform_SCM":
        train_transform = transforms.Compose([
            ProcessingCov_HyperLeaf2024_SCM(exclude_zero=True),
        ])
    
    if test_transform is None:
        test_transform = transforms.Compose([
            # Minimal preprocessing for test data
        ])
    elif test_transform == "CovTransform":
        test_transform = transforms.Compose([
            ProcessingCov_HyperLeaf2024(exclude_zero=True),
        ])
    elif test_transform == "CovTransform_SCM":
        test_transform = transforms.Compose([
            ProcessingCov_HyperLeaf2024_SCM(exclude_zero=True),
        ])
    
    # Create datasets
    train_dataset = HyperLeafDataset(
        data_dir=data_dir,
        split='train',
        task_type=task_type,
        transform=train_transform,
        target_columns=target_columns,
        max_samples=max_samples,
        images_dir=images_dir,
        label_noise_ratio=label_noise_ratio
    )
    if not(use_only_train_csv):
        test_dataset = HyperLeafDataset(
            data_dir=data_dir,
            split='test',
            task_type=task_type,
            transform=test_transform,
            target_columns=target_columns,
            max_samples=None,  # Don't limit test set
            images_dir=images_dir
        )
        train_size = int(len(train_dataset) * (1 - val_ratio))
        val_size = len(train_dataset) - train_size      
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    
    else:
        # Split train dataset into train and val
        train_size = int(len(train_dataset) * (1 - val_ratio - test_ratio))
        val_size = int(len(train_dataset) * val_ratio)
        test_size = len(train_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            train_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def get_num_classes_or_targets(task_type: str, target_columns: Optional[List[str]] = None) -> int:
    """
    Get the number of classes (classification) or number of targets (regression).
    
    Args:
        task_type: 'classification' or 'regression'
        target_columns: Optional list of target columns
    
    Returns:
        Number of classes or regression targets
    """
    if target_columns is None:
        if task_type == 'classification':
            return 4  # Heerup, Kvium, Rembrandt, Sheriff
        else:  # regression
            return 4  # GrainWeight, Gsw, PhiPS2, Fertilizer
    else:
        return len(target_columns)


if __name__ == "__main__":
    data_dir = "~/Documents/DATASET/HyperLeaf2024"
    
    print("=== Testing Classification Task ===")
    train_loader, val_loader, test_loader = create_hyperleaf_dataloaders(
        data_dir=data_dir,
        task_type='classification',
        batch_size=8,
        train_transform=transforms.Compose([
            ProcessingCov_HyperLeaf2024(exclude_zero=True),
        ]),
        test_transform=transforms.Compose([
            ProcessingCov_HyperLeaf2024(exclude_zero=True),
        ]),
        val_ratio=0.2,
        seed=42,
        num_workers=2,
        max_samples=100  # Limit to 100 samples per class
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Test a batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"\nTrain Batch {batch_idx}: images shape: {images.shape}, targets shape: {targets.shape}")
        print(f"Target example: {targets[0]}")
        if batch_idx >= 2:
            break
    
    print("\n=== Testing Regression Task ===")
    train_loader, val_loader, test_loader = create_hyperleaf_dataloaders(
        data_dir=data_dir,
        task_type='regression',
        batch_size=8,
        val_ratio=0.2,
        seed=42,
        num_workers=2
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Test a batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"\nTrain Batch {batch_idx}: images shape: {images.shape}, targets shape: {targets.shape}")
        print(f"Target example: {targets[0]}")
        if batch_idx >= 2:
            break
    
    print("\n=== Testing Test Set ===")
    for batch_idx, images in enumerate(test_loader):
        print(f"Test batch {batch_idx}: images shape: {images.shape}")
        if batch_idx >= 2:
            break
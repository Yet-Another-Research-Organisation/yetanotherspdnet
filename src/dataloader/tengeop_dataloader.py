#!/usr/bin/env python3
"""
TenGeoP-SARwv Dataset Loader for PyTorch
=========================================

Dataset loader for the TenGeoP-SARwv dataset containing 10 classes of oceanic/atmospheric phenomena
observed in SAR (Synthetic Aperture Radar) wave mode images.

Supported formats: TIFF (.tiff, .tif) and PNG (.png)

Classes:
- F: Pure Ocean Waves
- G: Wind Streaks  
- H: Micro Convective Cells
- I: Rain Cells
- J: Biological Slicks
- K: Sea Ice
- L: Iceberg
- M: Low Wind Area
- N: Atmospheric Front
- O: Oceanic Front

Author: AI Assistant
Date: 2025-10-07
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import argparse
from collections import Counter
import random
from pathlib import Path
import tifffile
import warnings
from torch.utils.data import random_split

warnings.filterwarnings("ignore", category=UserWarning)


class TenGeoSARDataset(Dataset):
    """
    PyTorch Dataset for TenGeoP-SARwv data
    
    Supports:
    - Automatic class distribution analysis
    - Balanced sampling with max samples per class
    - Flexible resizing (smallest size or fixed size)
    - TIFF and PNG image formats
    - TIFF image loading with proper normalization
    - Metadata extraction from filenames
    """
    
    # Class mapping from folder names to labels
    CLASS_NAMES = {
        'F': 'Pure Ocean Waves',
        'G': 'Wind Streaks', 
        'H': 'Micro Convective Cells',
        'I': 'Rain Cells',
        'J': 'Biological Slicks',
        'K': 'Sea Ice',
        'L': 'Iceberg',
        'M': 'Low Wind Area',
        'N': 'Atmospheric Front',
        'O': 'Oceanic Front'
    }
    
    def __init__(
        self,
        data_root: str,
        max_samples: Optional[int] = None,
        resize_mode: str = "min_size",  # "min_size" or "fixed_size"
        target_size: Optional[Tuple[int, int]] = None,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
        load_metadata: bool = True,
        channels: str = "single",  # "single" or "rgb" 
        verbose: bool = False
    ):
        """
        Initialize TenGeoSAR Dataset
        
        Args:
            data_root: Path to TenGeoP-SARwv folder
            max_samples: Maximum number of samples per class (None = all)
            resize_mode: "min_size" (resize to smallest image) or "fixed_size"
            target_size: (height, width) for fixed_size mode
            transform: Additional PyTorch transforms
            seed: Random seed for sampling
            load_metadata: Whether to load metadata from txt files
            channels: "single" for grayscale or "rgb" for 3-channel conversion
        """
        
        self.data_root = Path(data_root)
        self.max_samples = max_samples
        self.resize_mode = resize_mode
        self.target_size = target_size
        self.seed = seed
        self.channels = channels
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate paths
        if not self.data_root.exists():
            raise ValueError(f"Data root {data_root} does not exist")
            
        # Load class definitions and metadata
        # Temporary class mapping - will be rebuilt after filtering
        self._temp_class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.CLASS_NAMES.keys()))}
        
        if load_metadata:
            self.metadata = self._load_metadata()
        else:
            self.metadata = {}
            
        # Discover all TIFF files and build dataset
        self.samples = self._build_sample_list()
        
        # Rebuild class mappings based on actual classes present in samples
        actual_classes = sorted(set(sample[1] for sample in self.samples))
        if len(actual_classes) < len(self._temp_class_to_idx):
            # Some classes were filtered out, need to remap indices to be contiguous
            present_class_codes = sorted(set(
                cls_code for cls_code, idx in self._temp_class_to_idx.items()
                if any(sample[1] == idx for sample in self.samples)
            ))
            self.class_to_idx = {cls: new_idx for new_idx, cls in enumerate(present_class_codes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            # Remap sample labels to new contiguous indices
            old_to_new_idx = {old_idx: new_idx for new_idx, (cls, old_idx) in enumerate(
                (cls, self._temp_class_to_idx[cls]) for cls in present_class_codes
            )}
            self.samples = [
                (path, old_to_new_idx[old_idx], meta)
                for path, old_idx, meta in self.samples
            ]
            
            if self.verbose:
                print(f"\n⚠️  Some classes were filtered out due to insufficient samples")
                print(f"   Original classes: {len(self._temp_class_to_idx)}")
                print(f"   Remaining classes: {len(self.class_to_idx)}")
                print(f"   Classes kept: {list(self.class_to_idx.keys())}")
        else:
            # All classes present, use original mapping
            self.class_to_idx = self._temp_class_to_idx
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Calculate dataset statistics
        self.class_distribution = self._calculate_class_distribution()
        
        # Determine target size for resizing
        if self.resize_mode == "min_size" and self.target_size is None:
            self.target_size = self._find_min_size()
            
        # Setup transforms
        self.transform = self._build_transforms(transform)
        if self.verbose:
            print(f"Dataset initialized with {len(self.samples)} samples")
            print(f"Target size: {self.target_size}")
            print("Class distribution:")
            for cls, count in self.class_distribution.items():
                print(f"  {cls} ({self.CLASS_NAMES[cls]}): {count} samples")
        
    def _load_metadata(self) -> Dict:
        """Load metadata from txt files"""
        metadata = {}
        
        # Load class names
        class_file = self.data_root / "58682.txt"
        if class_file.exists():
            with open(class_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ' ' in line:
                        cls_code = line[0]
                        cls_name = line[2:]
                        # Verify consistency with hardcoded names
                        if cls_code in self.CLASS_NAMES:
                            assert self.CLASS_NAMES[cls_code] == cls_name
        
        # Load file metadata (coordinates, dates, etc.)
        meta_file = self.data_root / "58683.txt"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        file_path = parts[0]
                        cls_code = parts[1]
                        sensor = parts[2] 
                        timestamp = parts[3]
                        lon = float(parts[4])
                        lat = float(parts[5])
                        
                        # Extract filename without extension
                        filename = os.path.basename(file_path).replace('.png', '.tiff')
                        
                        metadata[filename] = {
                            'class': cls_code,
                            'sensor': sensor,
                            'timestamp': timestamp,
                            'longitude': lon,
                            'latitude': lat
                        }
        
        return metadata
    
    def _build_sample_list(self) -> List[Tuple[str, int, Dict]]:
        """Build list of all samples with paths and labels"""
        samples = []
        
        for class_folder in sorted(self.data_root.iterdir()):
            if not class_folder.is_dir():
                continue
                
            cls_code = class_folder.name
            if cls_code not in self._temp_class_to_idx:
                continue
                
            class_idx = self._temp_class_to_idx[cls_code]
            
            # Find all TIFF and PNG files in this class folder
            image_files = (list(class_folder.glob("*.tiff")) + 
                          list(class_folder.glob("*.tif")) +
                          list(class_folder.glob("*.png")) +
                          list(class_folder.glob("*.PNG")))
            
            # Apply sampling if max_samples is specified
            if self.max_samples is not None:
                if len(image_files) > self.max_samples:
                    # Downsample to the requested maximum
                    image_files = random.sample(image_files, self.max_samples)
                else:
                    # If the class has fewer samples than max_samples, keep all available samples.
                    # Previously some logic skipped the class entirely when it had fewer than
                    # max_samples; that caused datasets to be empty when max_samples was set
                    # to a very large value. We now keep the class and optionally warn.
                    if self.verbose:
                        print(f"⚠️  Class {cls_code} ({self.CLASS_NAMES[cls_code]}) has only {len(image_files)} samples (< requested max_samples={self.max_samples}); keeping available samples.")
            
            for image_file in image_files:
                # Get metadata for this file if available
                file_meta = self.metadata.get(image_file.name, {})
                
                samples.append((str(image_file), class_idx, file_meta))
        
        return samples
    
    def _calculate_class_distribution(self) -> Dict[str, int]:
        """Calculate number of samples per class"""
        distribution = {}
        for _, class_idx, _ in self.samples:
            cls_code = self.idx_to_class[class_idx]
            distribution[cls_code] = distribution.get(cls_code, 0) + 1
        return distribution
    
    def _find_min_size(self) -> Tuple[int, int]:
        """Find the minimum size across all images for consistent resizing"""
        min_height, min_width = float('inf'), float('inf')
        
        print("Scanning images to find minimum size...")
        sample_size = min(50, len(self.samples))  # Sample subset for efficiency
        sample_indices = random.sample(range(len(self.samples)), sample_size)
        
        for idx in sample_indices:
            file_path, _, _ = self.samples[idx]
            try:
                # Check file extension to determine how to load
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.tiff', '.tif']:
                    with tifffile.TiffFile(file_path) as tif:
                        height, width = tif.pages[0].shape
                elif file_ext in ['.png']:
                    with Image.open(file_path) as img:
                        width, height = img.size
                else:
                    continue
                    
                min_height = min(min_height, height)
                min_width = min(min_width, width)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        if min_height == float('inf'):
            # Fallback if no images could be read
            min_height, min_width = 256, 256
            
        print(f"Minimum size found: {min_height}x{min_width}")
        return (min_height, min_width)
    
    def _build_transforms(self, additional_transform: Optional[transforms.Compose]) -> transforms.Compose:
        """Build transformation pipeline"""
        transform_list = []
        
        # Resize if needed
        if self.target_size:
            transform_list.append(transforms.Resize(self.target_size))
        
        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        
        # Add channel dimension handling
        if self.channels == "rgb":
            # Convert single channel to RGB by repeating
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        
        # Normalize to [0, 1] range (TIFF might have different ranges)
        transform_list.append(transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)))
        
        # Add user transforms
        if additional_transform:
            transform_list.append(additional_transform)
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample from the dataset
        
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer class label
            metadata: Dictionary with file metadata
        """
        file_path, label, metadata = self.samples[idx]
        
        # Load image (TIFF or PNG)
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.tiff', '.tif']:
                # Load TIFF with tifffile
                image_array = tifffile.imread(file_path)
            elif file_ext in ['.png']:
                # Load PNG with PIL
                with Image.open(file_path) as img:
                    image_array = np.array(img)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Handle different image formats
            if image_array.ndim == 2:
                # Grayscale
                image = Image.fromarray(image_array.astype(np.float32), mode='F')
            elif image_array.ndim == 3 and image_array.shape[2] == 1:
                # Single channel
                image = Image.fromarray(image_array.squeeze().astype(np.float32), mode='F')
            elif image_array.ndim == 3 and image_array.shape[2] == 3:
                # RGB
                if image_array.dtype == np.uint8 or image_array.max() <= 255:
                    image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
                else:
                    # Normalize to uint8 range if values are larger
                    normalized = (image_array / image_array.max() * 255).astype(np.uint8)
                    image = Image.fromarray(normalized, mode='RGB')
            elif image_array.ndim == 3 and image_array.shape[2] == 4:
                # RGBA - convert to RGB
                if image_array.dtype == np.uint8 or image_array.max() <= 255:
                    rgba_img = Image.fromarray(image_array.astype(np.uint8), mode='RGBA')
                    image = rgba_img.convert('RGB')
                else:
                    normalized = (image_array / image_array.max() * 255).astype(np.uint8)
                    rgba_img = Image.fromarray(normalized, mode='RGBA')
                    image = rgba_img.convert('RGB')
            else:
                raise ValueError(f"Unexpected image shape: {image_array.shape}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a black image as fallback
            if self.target_size:
                height, width = self.target_size
            else:
                height, width = 256, 256
            image = Image.fromarray(np.zeros((height, width), dtype=np.float32), mode='F')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced loss computation"""
        counts = list(self.class_distribution.values())
        total = sum(counts)
        weights = [total / (len(counts) * count) for count in counts]
        return torch.FloatTensor(weights)
    
    def save_dataset(self, save_path: str):
        """Save the prepared dataset to .pt format"""
        print(f"Saving dataset to {save_path}...")
        
        # Prepare data for saving
        data = {
            'samples': [],
            'labels': [],
            'metadata': [],
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'class_names': self.CLASS_NAMES,
            'class_distribution': self.class_distribution,
            'target_size': self.target_size,
            'num_classes': len(self.class_to_idx)
        }
        
        # Load all samples into memory
        for i in range(len(self)):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(self)}")
            
            image, label, metadata = self[i]
            data['samples'].append(image)
            data['labels'].append(label)
            data['metadata'].append(metadata)
        
        # Convert to tensors
        data['samples'] = torch.stack(data['samples'])
        data['labels'] = torch.LongTensor(data['labels'])
        
        # Save to file
        torch.save(data, save_path)
        print(f"Dataset saved successfully! Shape: {data['samples'].shape}")


def create_dataloader(
    data_root: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    resize_mode: str = "min_size",
    target_size: Optional[Tuple[int, int]] = None,
    split_ratio: float = 0.8,
    val_ratio: float = 0.2,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for TenGeoP-SARwv dataset
    
    Args:
        data_root: Path to dataset
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes
        max_samples: Maximum samples per class
        resize_mode: Resize strategy
        target_size: Target image size
        split_ratio: Train+Val/Test split ratio (default: 0.8)
        val_ratio: Validation split from Train+Val (default: 0.2)
        **kwargs: Additional arguments for dataset
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    
    dataset = TenGeoSARDataset(
        data_root=data_root,
        max_samples=max_samples,
        resize_mode=resize_mode,
        target_size=target_size,
        **kwargs
    )

    if split_ratio is not None and 0 < split_ratio < 1:
        train_size = int(len(dataset) * split_ratio)
        test_size = len(dataset) - train_size
        val_size = int(train_size * val_ratio)
        train_size -= val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(dataset.seed)
        )
    
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=torch.Generator().manual_seed(dataset.seed)
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=torch.Generator().manual_seed(dataset.seed)
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=torch.Generator().manual_seed(dataset.seed)
        )
        return train_dataloader, val_dataloader, test_dataloader



if __name__ == "__main__":
    path = "~/Documents/Dataset/BATCHNORM_dataset/TenGeoP-SARwv"

    train_dataloader, test_dataloader = create_dataloader(
        data_root=path,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        max_samples=500,
        resize_mode="min_size",
        target_size=None,
        seed=42,
        load_metadata=True,
        channels="single",
        verbose=True
    )

    for i, (images, labels, metadata) in enumerate(train_dataloader):
        print(f"Batch {i+1}:")
        print(f"  Shape des images: {images.shape}")
        print(f"  Shape des labels: {labels.shape}")
        print(f"  Labels dans ce batch: {labels.tolist()}")

        
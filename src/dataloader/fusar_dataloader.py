#!/usr/bin/env python3
"""
FUSAR-Ship Dataset Loader for PyTorch
=====================================

Dataset loader for the FUSAR-Ship dataset containing ship classification data
from SAR (Synthetic Aperture Radar) images.

Classification Levels:
- basic: Main folders as classes (Cargo, Tanker, Passenger, etc.)
- fine: Sub-folders as classes (BulkCarrier, ContainerShip, etc.)

Supported formats: TIFF (.tiff, .tif) and PNG (.png)

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
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import tifffile
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class FUSARShipDataset(Dataset):
    """
    PyTorch Dataset for FUSAR-Ship data
    
    Supports:
    - Two classification levels (basic/fine)
    - Automatic class distribution analysis
    - Balanced sampling with max samples per class
    - Flexible resizing (smallest size or fixed size)
    - TIFF and PNG image formats
    - Metadata extraction from CSV files
    """
    
    def __init__(
        self,
        data_root: str,
        class_level: str = "basic",  # "basic" or "fine"
        max_samples: Optional[int] = None,
        resize_mode: str = "min_size",  # "min_size" or "fixed_size"
        target_size: Optional[Tuple[int, int]] = None,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
        load_metadata: bool = True,
        channels: str = "single",  # "single" or "rgb"
        verbose: bool = False,
        min_samples: Optional[int] = None  # Minimum samples required per class (drop classes below this threshold)
    ):
        """
        Initialize FUSAR-Ship Dataset
        
        Args:
            data_root: Path to FUSAR-Ship1.0 folder
            class_level: "basic" (main folders) or "fine" (sub-folders)
            max_samples: Maximum number of samples per class (None = all)
            resize_mode: "min_size" (resize to smallest image) or "fixed_size"
            target_size: (height, width) for fixed_size mode
            transform: Additional PyTorch transforms
            seed: Random seed for sampling
            load_metadata: Whether to load metadata from CSV files
            channels: "single" for grayscale or "rgb" for 3-channel conversion
            verbose: Whether to print dataset info
            min_samples: Minimum number of samples required per class (drop classes below this threshold)
        """
        
        self.data_root = Path(data_root)
        self.class_level = class_level
        self.max_samples = max_samples
        self.resize_mode = resize_mode
        self.target_size = target_size
        self.seed = seed
        self.channels = channels
        self.verbose = verbose
        self.min_samples = min_samples
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate paths
        if not self.data_root.exists():
            raise ValueError(f"Data root {data_root} does not exist")
            
        if class_level not in ["basic", "fine"]:
            raise ValueError(f"class_level must be 'basic' or 'fine', got '{class_level}'")
            
        # Load metadata if available
        if load_metadata:
            self.metadata = self._load_metadata()
        else:
            self.metadata = {}
            
        # Build class mappings based on classification level
        self.class_names, self.class_to_idx, self.idx_to_class = self._build_class_mappings()
        
        # Discover all image files and build dataset
        self.samples = self._build_sample_list()
        
        # Calculate dataset statistics
        self.class_distribution = self._calculate_class_distribution()
        
        # Filter classes if min_samples is specified
        if self.min_samples is not None:
            self._filter_classes_by_min_samples()
        
        # Determine target size for resizing
        if self.resize_mode == "min_size" and self.target_size is None:
            self.target_size = self._find_min_size()
            
        # Setup transforms
        self.transform = self._build_transforms(transform)
        if self.verbose:
            print(f"FUSAR-Ship Dataset initialized:")
            print(f"  Classification level: {self.class_level}")
            print(f"  Total samples: {len(self.samples)}")
            print(f"  Number of classes: {len(self.class_to_idx)}")
            print(f"  Target size: {self.target_size}")
            print(f"  Min samples per class: {self.min_samples or 'None'}")
            print(f"Class distribution:")
            for cls, count in sorted(self.class_distribution.items()):
                print(f"  {cls}: {count} samples")
        
    def _filter_classes_by_min_samples(self):
        """Filter out classes with fewer samples than min_samples"""
        if self.min_samples is None:
            return
        
        # Find classes to keep
        classes_to_keep = set()
        for cls_key, count in self.class_distribution.items():
            if count >= self.min_samples:
                classes_to_keep.add(self.class_to_idx[cls_key])
        
        if self.verbose and len(classes_to_keep) < len(self.class_to_idx):
            dropped_classes = len(self.class_to_idx) - len(classes_to_keep)
            print(f"\nDropping {dropped_classes} classes with fewer than {self.min_samples} samples")
            for cls_key, count in sorted(self.class_distribution.items()):
                if count < self.min_samples:
                    print(f"  Dropped: {cls_key} ({count} samples)")
        
        # Filter samples
        original_count = len(self.samples)
        self.samples = [
            (path, label, meta) for path, label, meta in self.samples
            if label in classes_to_keep
        ]
        
        # Rebuild class mappings for remaining classes
        old_idx_to_class = self.idx_to_class.copy()
        remaining_classes = sorted([
            old_idx_to_class[idx] for idx in classes_to_keep
        ])
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(remaining_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Update class_names to only include remaining classes
        self.class_names = {
            cls: self.class_names[cls] for cls in remaining_classes
        }
        
        # Remap labels in samples
        old_to_new_idx = {
            old_idx: self.class_to_idx[old_idx_to_class[old_idx]]
            for old_idx in classes_to_keep
        }
        
        self.samples = [
            (path, old_to_new_idx[label], meta)
            for path, label, meta in self.samples
        ]
        
        # Recalculate class distribution
        self.class_distribution = self._calculate_class_distribution()
        
        if self.verbose:
            print(f"  Kept {len(self.samples)}/{original_count} samples across {len(self.class_to_idx)} classes")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from CSV files"""
        metadata = {}
        
        # Check for meta.csv
        meta_csv = self.data_root / "meta.csv"
        if meta_csv.exists():
            try:
                df = pd.read_csv(meta_csv)
                for _, row in df.iterrows():
                    # Assuming CSV has columns like 'filename', 'ship_type', etc.
                    if 'filename' in df.columns:
                        filename = row['filename']
                        metadata[filename] = row.to_dict()
            except Exception as e:
                print(f"Warning: Could not load metadata from {meta_csv}: {e}")
        
        return metadata
    
    def _build_class_mappings(self) -> Tuple[Dict[str, str], Dict[str, int], Dict[int, str]]:
        """Build class mappings based on classification level"""
        class_names = {}
        
        if self.class_level == "basic":
            # Use main folders as classes
            for main_folder in sorted(self.data_root.iterdir()):
                if main_folder.is_dir() and not main_folder.name.startswith('.'):
                    class_names[main_folder.name] = main_folder.name
                    
        elif self.class_level == "fine":
            # Use sub-folders as classes
            for main_folder in sorted(self.data_root.iterdir()):
                if main_folder.is_dir() and not main_folder.name.startswith('.'):
                    for sub_folder in sorted(main_folder.iterdir()):
                        if sub_folder.is_dir() and not sub_folder.name.startswith('.'):
                            # Create unique class name: MainFolder_SubFolder
                            class_key = f"{main_folder.name}_{sub_folder.name}"
                            class_names[class_key] = f"{main_folder.name} - {sub_folder.name}"
        
        # Create mappings
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names.keys()))}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        return class_names, class_to_idx, idx_to_class
    
    def _build_sample_list(self) -> List[Tuple[str, int, Dict]]:
        """Build list of all samples with paths and labels"""
        samples = []
        
        for main_folder in sorted(self.data_root.iterdir()):
            if not main_folder.is_dir() or main_folder.name.startswith('.'):
                continue
                
            if self.class_level == "basic":
                # Images directly in main folder or any sub-folder belong to main class
                class_key = main_folder.name
                if class_key not in self.class_to_idx:
                    continue
                class_idx = self.class_to_idx[class_key]
                
                # Collect images from main folder and all sub-folders
                image_files = []
                for root, dirs, files in os.walk(main_folder):
                    for file in files:
                        if file.lower().endswith(('.tiff', '.tif', '.png')):
                            image_files.append(Path(root) / file)
                
                # Apply sampling if max_samples is specified
                if self.max_samples and len(image_files) > self.max_samples:
                    image_files = random.sample(image_files, self.max_samples)
                
                for image_file in image_files:
                    file_meta = self.metadata.get(image_file.name, {})
                    samples.append((str(image_file), class_idx, file_meta))
                    
            elif self.class_level == "fine":
                # Each sub-folder is a separate class
                for sub_folder in sorted(main_folder.iterdir()):
                    if not sub_folder.is_dir() or sub_folder.name.startswith('.'):
                        continue
                        
                    class_key = f"{main_folder.name}_{sub_folder.name}"
                    if class_key not in self.class_to_idx:
                        continue
                    class_idx = self.class_to_idx[class_key]
                    
                    # Find all image files in this sub-folder
                    image_files = []
                    for root, dirs, files in os.walk(sub_folder):
                        for file in files:
                            if file.lower().endswith(('.tiff', '.tif', '.png')):
                                image_files.append(Path(root) / file)
                    
                    # Apply sampling if max_samples is specified
                    if self.max_samples and len(image_files) > self.max_samples:
                        image_files = random.sample(image_files, self.max_samples)
                    
                    for image_file in image_files:
                        file_meta = self.metadata.get(image_file.name, {})
                        samples.append((str(image_file), class_idx, file_meta))
        
        return samples
    
    def _calculate_class_distribution(self) -> Dict[str, int]:
        """Calculate number of samples per class"""
        distribution = {}
        for _, class_idx, _ in self.samples:
            cls_key = self.idx_to_class[class_idx]
            distribution[cls_key] = distribution.get(cls_key, 0) + 1
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
    
    def get_class_info(self) -> Dict:
        """Get detailed class information"""
        class_info = {}
        for cls_key, cls_idx in self.class_to_idx.items():
            class_info[cls_key] = {
                'index': cls_idx,
                'name': self.class_names[cls_key],
                'count': self.class_distribution.get(cls_key, 0),
                'level': self.class_level
            }
        return class_info
    
    def print_dataset_summary(self):
        """Print a comprehensive dataset summary"""
        print(f"\n{'='*60}")
        print(f"FUSAR-Ship Dataset Summary")
        print(f"{'='*60}")
        print(f"Classification Level: {self.class_level}")
        print(f"Total Samples: {len(self.samples):,}")
        print(f"Number of Classes: {len(self.class_to_idx)}")
        print(f"Image Target Size: {self.target_size}")
        print(f"Channels: {self.channels}")
        print(f"Max Samples per Class: {self.max_samples or 'No limit'}")
        
        print(f"\nClass Distribution:")
        print(f"{'-'*60}")
        total_samples = sum(self.class_distribution.values())
        for cls_key in sorted(self.class_distribution.keys()):
            count = self.class_distribution[cls_key]
            percentage = (count / total_samples) * 100
            class_name = self.class_names[cls_key]
            print(f"{cls_key:25} | {count:6,} samples ({percentage:5.1f}%) | {class_name}")
        
        # Class balance analysis
        counts = list(self.class_distribution.values())
        min_count, max_count = min(counts), max(counts)
        balance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nDataset Balance Analysis:")
        print(f"  Min samples per class: {min_count:,}")
        print(f"  Max samples per class: {max_count:,}")
        print(f"  Balance ratio (max/min): {balance_ratio:.2f}")
        
        if balance_ratio < 2.0:
            print("  ✅ Well balanced dataset")
        elif balance_ratio < 5.0:
            print("  ⚠️ Moderately imbalanced dataset")
        else:
            print("  ❌ Highly imbalanced dataset - consider using class weights")


def create_fusar_dataloader(
    data_root: str,
    class_level: str = "basic",
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
    Create DataLoaders for FUSAR-Ship dataset
    
    Args:
        data_root: Path to FUSAR-Ship1.0 dataset
        class_level: "basic" (main folders) or "fine" (sub-folders)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes
        max_samples: Maximum samples per class
        resize_mode: Resize strategy
        target_size: Target image size
        split_ratio: Train+Val/Test split ratio (default: 0.8)
        val_ratio: Validation split from Train+Val (default: 0.2)
        **kwargs: Additional arguments for dataset (including min_samples)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    
    dataset = FUSARShipDataset(
        data_root=data_root,
        class_level=class_level,
        max_samples=max_samples,
        resize_mode=resize_mode,
        target_size=target_size,
        **kwargs
    )
    
    if split_ratio is not None and 0.0 < split_ratio < 1.0:
        # Split dataset into train+val and test
        total_size = len(dataset)
        train_val_size = int(total_size * split_ratio)
        test_size = total_size - train_val_size
        
        train_val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_val_size, test_size],
            generator=torch.Generator().manual_seed(dataset.seed)
        )
        
        # Split train+val into train and val
        val_size = int(train_val_size * val_ratio)
        train_size = train_val_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(dataset.seed)
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
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        return dataloader, None, None



if __name__ == "__main__":
    data_root = '~/Documents/Dataset/BATCHNORM_dataset/FUSAR_Ship1.0'
    train_dataloader, val_dataloader, test_dataloader = create_fusar_dataloader(
        data_root=data_root,
        class_level="fine",
        batch_size=32,
        max_samples=100,
        resize_mode="min_size",
        target_size=None,
        channels="rgb",
        split_ratio=0.8,
        val_ratio=0.2,
        seed=42,
        min_samples=50,  # Drop classes with fewer than 50 samples
        verbose=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataloader.dataset)} samples")
    print(f"  Val: {len(val_dataloader.dataset)} samples")
    print(f"  Test: {len(test_dataloader.dataset)} samples")
    
    for i, (images, labels, metadata) in enumerate(train_dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Shape des images: {images.shape}")
        print(f"  Shape des labels: {labels.shape}")
        print(f"  Labels dans ce batch: {labels.tolist()}")
        if i == 2:  # Juste pour afficher les 3 premiers lots   
            break
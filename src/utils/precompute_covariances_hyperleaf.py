#!/usr/bin/env python3
"""
Script to precompute covariance matrices for HyperLeaf dataset
Converts all TIFF images to covariance matrices and saves them as .pt files

Usage:
    python precompute_covariances.py --data_dir ~/Documents/Dataset/BATCHNORM_dataset/HyperLeaf2024
    python precompute_covariances.py --data_dir ~/Documents/Dataset/BATCHNORM_dataset/HyperLeaf2024 --num_workers 8
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import tifffile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Add parent directory to path
sys.path.append("../")

from dataloader.hyperleaf_dataloader import ProcessingCov_HyperLeaf2024_SCM


def load_hyperspectral_image(image_path: str) -> torch.Tensor:
    """
    Load a hyperspectral TIFF image and convert to tensor.
    
    Args:
        image_path: Path to the TIFF file
        
    Returns:
        Tensor of shape (H, W, C) where C is the number of bands
    """
    try:
        # Load using tifffile for better support of scientific TIFFs
        image_array = tifffile.imread(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    # Convert to float32 and normalize
    image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1] range if needed
    if image_array.max() > 1.0:
        # image_array = image_array / 65535.0  # Assuming 16-bit images
        image_array = image_array / 1640.9 # Using dataset-specific normalization factor for eigenvalue stability (geometric mean ~1.0)
    
    # Change shape from (C, H, W) to (H, W, C)
    image_array = np.moveaxis(image_array, 0, -1)
    
    # Convert to tensor
    image_tensor = torch.tensor(image_array)
    
    return image_tensor


def process_single_image(args_tuple):
    """
    Process a single image: load, compute covariance, and save.
    This function is used for parallel processing.
    
    Args:
        args_tuple: Tuple of (image_path, output_path, transform)
        
    Returns:
        Tuple of (image_id, success, error_message)
    """
    image_path, output_path, exclude_zero = args_tuple
    image_id = Path(image_path).stem
    
    try:
        # Load image
        image = load_hyperspectral_image(image_path)
        
        # Create transform
        transform = ProcessingCov_HyperLeaf2024_SCM(exclude_zero=exclude_zero)
        
        # Compute covariance matrix
        cov_matrix = transform(image)
        
        # Save as PyTorch tensor
        torch.save(cov_matrix, output_path)
        
        return (image_id, True, None)
        
    except Exception as e:
        return (image_id, False, str(e))


def precompute_covariances(data_dir: str, 
                          exclude_zero: bool = True,
                          num_workers: int = None,
                          force_recompute: bool = False):
    """
    Precompute all covariance matrices for HyperLeaf dataset.
    
    Args:
        data_dir: Root directory of HyperLeaf dataset
        exclude_zero: Whether to exclude zero pixels (default: True)
        num_workers: Number of parallel workers (default: CPU count)
        force_recompute: If True, recompute even if covariance already exists
    """
    data_dir = os.path.expanduser(data_dir)
    images_dir = os.path.join(data_dir, 'images')
    cov_dir = os.path.join(data_dir, 'cov')
    
    # Validate input directory
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Create output directory
    os.makedirs(cov_dir, exist_ok=True)
    print(f"Created/verified covariance directory: {cov_dir}")
    
    # Get list of all TIFF images
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tiff') or f.endswith('.tif')])
    
    if len(image_files) == 0:
        raise ValueError(f"No TIFF images found in {images_dir}")
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Exclude zero pixels: {exclude_zero}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(image_files))
    
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    tasks = []
    skipped = 0
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_id = Path(image_file).stem
        output_path = os.path.join(cov_dir, f"{image_id}.pt")
        
        # Skip if already exists and not forcing recompute
        if os.path.exists(output_path) and not force_recompute:
            skipped += 1
            continue
        
        tasks.append((image_path, output_path, exclude_zero))
    
    if skipped > 0:
        print(f"Skipping {skipped} already computed covariances (use --force to recompute)")
    
    if len(tasks) == 0:
        print("\nAll covariances already computed!")
        return
    
    print(f"\nProcessing {len(tasks)} images...")
    
    # Process images in parallel
    successful = 0
    failed = []
    
    if num_workers == 1:
        # Sequential processing for debugging
        for task in tqdm(tasks, desc="Computing covariances"):
            image_id, success, error = process_single_image(task)
            if success:
                successful += 1
            else:
                failed.append((image_id, error))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc="Computing covariances") as pbar:
                for future in as_completed(futures):
                    image_id, success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed.append((image_id, error))
                    pbar.update(1)
    
    # Print summary
    print("\n" + "="*60)
    print("PRECOMPUTATION COMPLETE")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Already computed: {skipped}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Covariances saved in: {cov_dir}")
    
    if failed:
        print("\nFailed images:")
        for image_id, error in failed[:10]:  # Show first 10 failures
            print(f"  - {image_id}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Print example of saved covariance
    if successful > 0:
        example_file = os.path.join(cov_dir, f"{Path(tasks[0][0]).stem}.pt")
        if os.path.exists(example_file):
            example_cov = torch.load(example_file)
            print(f"\nExample covariance matrix shape: {example_cov.shape}")
            print(f"Example covariance matrix dtype: {example_cov.dtype}")
            print(f"File size: {os.path.getsize(example_file) / 1024:.2f} KB")
    
    print("="*60)


def verify_covariances(data_dir: str, num_samples: int = 5):
    """
    Verify that precomputed covariances are valid.
    
    Args:
        data_dir: Root directory of HyperLeaf dataset
        num_samples: Number of samples to verify
    """
    data_dir = os.path.expanduser(data_dir)
    cov_dir = os.path.join(data_dir, 'cov')
    
    if not os.path.exists(cov_dir):
        print(f"Covariance directory not found: {cov_dir}")
        return
    
    cov_files = sorted([f for f in os.listdir(cov_dir) if f.endswith('.pt')])
    
    if len(cov_files) == 0:
        print(f"No covariance files found in {cov_dir}")
        return
    
    print(f"\nVerifying {min(num_samples, len(cov_files))} covariance matrices...")
    print("-" * 60)
    
    for i, cov_file in enumerate(cov_files[:num_samples]):
        cov_path = os.path.join(cov_dir, cov_file)
        
        try:
            cov = torch.load(cov_path)
            
            # Check properties
            is_square = cov.shape[0] == cov.shape[1]
            is_symmetric = torch.allclose(cov, cov.T, rtol=1e-5, atol=1e-7)
            
            # Check positive semi-definite (eigenvalues should be non-negative)
            eigenvalues = torch.linalg.eigvalsh(cov)
            min_eig = eigenvalues.min().item()
            is_psd = min_eig >= -1e-6  # Allow small numerical errors
            
            print(f"{i+1}. {cov_file}")
            print(f"   Shape: {cov.shape}")
            print(f"   Dtype: {cov.dtype}")
            print(f"   Square: {is_square}")
            print(f"   Symmetric: {is_symmetric}")
            print(f"   Positive Semi-Definite: {is_psd} (min eigenvalue: {min_eig:.2e})")
            print(f"   Trace: {torch.trace(cov).item():.6f}")
            print(f"   Condition number: {eigenvalues.max().item() / (eigenvalues.min().item() + 1e-10):.2e}")
            
            if not (is_square and is_symmetric and is_psd):
                print(f"   ⚠️  WARNING: Matrix has invalid properties!")
            else:
                print(f"   ✓ Valid covariance matrix")
            print()
            
        except Exception as e:
            print(f"{i+1}. {cov_file}")
            print(f"   ✗ ERROR: {e}")
            print()
    
    print("-" * 60)
    print(f"Total covariance files: {len(cov_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute covariance matrices for HyperLeaf dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Root directory of HyperLeaf dataset (e.g., ~/Documents/Dataset/BATCHNORM_dataset/HyperLeaf2024)'
    )
    
    parser.add_argument(
        '--exclude_zero',
        action='store_true',
        default=True,
        help='Exclude zero pixels from covariance computation'
    )
    
    parser.add_argument(
        '--no_exclude_zero',
        dest='exclude_zero',
        action='store_false',
        help='Include zero pixels in covariance computation'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect CPU count)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation even if covariances already exist'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify precomputed covariances after computation'
    )
    
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing covariances without computing'
    )
    
    parser.add_argument(
        '--verify_samples',
        type=int,
        default=5,
        help='Number of samples to verify'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("HyperLeaf Covariance Precomputation")
    print("="*60)
    
    if args.verify_only:
        # Only verify
        verify_covariances(args.data_dir, args.verify_samples)
    else:
        # Compute covariances
        precompute_covariances(
            data_dir=args.data_dir,
            exclude_zero=args.exclude_zero,
            num_workers=args.num_workers,
            force_recompute=args.force
        )
        
        # Verify if requested
        if args.verify:
            print("\n")
            verify_covariances(args.data_dir, args.verify_samples)


if __name__ == "__main__":
    main()

"""
Plotting utilities for training and evaluation visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple


def setup_matplotlib(serif=False):
    """Setup matplotlib with better defaults"""
    if serif:
        plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["EB Garamond", "Times New Roman", "serif"],
            "text.usetex": False,  # Disable LaTeX rendering to use system fonts
            "mathtext.fontset": "dejavusans",  # For math symbols
        }
    )
    else:
        plt.rcParams.update(
        {
            "font.family": "sans-serif",  # Use sans-serif family
            "font.sans-serif": ["Fira Sans", "DejaVu Sans", "Arial"],  # Fira Sans as priority
            "text.usetex": False,  # Disable LaTeX rendering to use system fonts
            "mathtext.fontset": "dejavusans",  # For math symbols
        }
    )
    plt.style.use('seaborn-v0_8-paper')
    # plt.style.use('seaborn-colorblind')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10




def plot_loss_acc_lr(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    learning_rates: Optional[List[float]] = None,
    model: Optional[str] = None,
):
    """
    Plot training/validation loss, accuracy, and learning rate over epochs
    
    Parameters
    ----------
    train_losses : list
        Training losses per epoch
    train_accs : list
        Training accuracies per epoch
    val_losses : list, optional
        Validation losses per epoch
    val_accs : list, optional
        Validation accuracies per epoch
    learning_rates : list, optional
        Learning rates per epoch
    model : str, optional
        Model description to display
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    has_val = val_losses is not None and val_accs is not None
    has_lr = learning_rates is not None
    
    # Determine grid layout
    if has_lr:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if has_val:
        ax_loss.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax_acc = axes[1]
    ax_acc.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=4)
    if has_val:
        ax_acc.plot(epochs, val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=4)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Training and Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    # Plot Learning Rate
    if has_lr:
        ax_lr = axes[2]
        ax_lr.plot(epochs, learning_rates, 'g-^', linewidth=2, markersize=4)
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('Learning Rate Schedule')
        ax_lr.set_yscale('log')
        ax_lr.grid(True, alpha=0.3)
    
    # # Add model info if provided
    # if model:
    #     fig.text(0.02, 0.02, f'Model: {model}', fontsize=8, 
    #             verticalalignment='bottom', wrap=True)
    
    plt.tight_layout()
    return fig


def plot_evaluation_metrics(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    per_class_f1: Dict[str, float],
    metrics_summary: Dict[str, float],
    title: str = "Evaluation Results",
):
    """
    Create a comprehensive evaluation figure with confusion matrix, 
    per-class F1 scores, and summary metrics
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix (n_classes x n_classes)
    class_names : list
        List of class names
    per_class_f1 : dict
        Dictionary mapping class names to F1 scores
    metrics_summary : dict
        Dictionary with overall metrics (loss, accuracy, precision, recall, f1)
    title : str
        Overall title for the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # 1. Confusion Matrix (top-left, larger)
    ax_cm = fig.add_subplot(gs[0, 0])
    
    # Normalize confusion matrix for better visualization
    # Handle division by zero by replacing NaN with 0
    row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    cm_normalized = confusion_matrix.astype('float') / row_sums
    
    im = ax_cm.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax_cm.set_title('Normalized Confusion Matrix', fontsize=14, pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)
    
    # Axis labels
    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_ylabel('True Label', fontsize=12)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax_cm.text(j, i, f'{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.2f})',
                      ha="center", va="center",
                      color="white" if cm_normalized[i, j] > thresh else "black",
                      fontsize=8)
    
    # 2. Per-Class F1 Scores (top-right)
    ax_f1 = fig.add_subplot(gs[0, 1])
    
    # Sort classes by F1 score
    sorted_items = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
    sorted_classes = [item[0] for item in sorted_items]
    sorted_f1 = [item[1] for item in sorted_items]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_classes))
    bars = ax_f1.barh(y_pos, sorted_f1, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Color bars based on F1 score
    for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
        if score >= 0.8:
            bar.set_color('green')
        elif score >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax_f1.set_yticks(y_pos)
    ax_f1.set_yticklabels(sorted_classes, fontsize=9)
    ax_f1.set_xlabel('F1 Score', fontsize=12)
    ax_f1.set_title('Per-Class F1 Scores', fontsize=14, pad=10)
    ax_f1.set_xlim([0, 1.0])
    ax_f1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sorted_f1):
        ax_f1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def analyze_covariance_matrices(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Dict[int, str],
    output_dir: str,
    n_samples_per_class: int = 1,
    dtype: torch.dtype = torch.float32,
    save_format: str = 'png',
) -> Tuple[str, str, str, Dict[str, float]]:
    """
    Analyze covariance matrices at SPDNet input and before LogEig layer.
    Returns a text report, paths to output directories with individual figures,
    and conditioning statistics.
    
    Parameters
    ----------
    model : nn.Module
        The model to analyze
    data_loader : DataLoader
        Data loader for samples
    device : torch.device
        Device to use
    class_names : Dict[int, str]
        Mapping from class index to class name
    output_dir : str
        Directory to save outputs
    n_samples_per_class : int
        Number of samples per class to analyze.
        If -1, analyze all available samples (for conditioning stats only, no figures saved)
    dtype : torch.dtype
        Data type for model inputs
    save_format : str
        Format to save figures ('png' or 'pdf'), default 'png'
    
    Returns
    -------
    Tuple[str, str, str, Dict[str, float]]
        (report_text, input_cov_dir, output_cov_dir, conditioning_stats)
        conditioning_stats contains: 
            - 'input_cond_mean': mean condition number at input
            - 'input_cond_std': std of condition number at input
            - 'output_cond_mean': mean condition number before LogEig
            - 'output_cond_std': std of condition number before LogEig
    """
    import os
    
    model.eval()
    
    # Validate save format
    if save_format not in ['png', 'pdf']:
        raise ValueError(f"save_format must be 'png' or 'pdf', got '{save_format}'")
    
    # If n_samples_per_class is -1, we analyze all samples but don't save figures
    save_figures = (n_samples_per_class > 0)
    analyze_all = (n_samples_per_class == -1)
    
    # Create output directories only if saving figures
    if save_figures:
        input_cov_dir = os.path.join(output_dir, "input_covariance")
        output_cov_dir = os.path.join(output_dir, "output_covariance")
        os.makedirs(input_cov_dir, exist_ok=True)
        os.makedirs(output_cov_dir, exist_ok=True)
    else:
        input_cov_dir = ""
        output_cov_dir = ""
    
    # Storage for covariance matrices by class
    cov_input = {class_idx: [] for class_idx in class_names.keys()}
    cov_before_logeig = {class_idx: [] for class_idx in class_names.keys()}
    
    # Storage for condition numbers
    input_condition_numbers = []
    output_condition_numbers = []
    
    # Hook to capture matrices before LogEig
    matrices_before_logeig = []
    
    def hook_before_logeig(module, input, output):
        matrices_before_logeig.append(input[0].detach().cpu())
    
    # Register hook on LogEig layer
    hook_handle = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LogEig':
            hook_handle = module.register_forward_hook(hook_before_logeig)
            break
    
    # Collect data
    samples_per_class = {class_idx: 0 for class_idx in class_names.keys()}
    
    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 2:
                inputs, labels = batch_data
            else:
                inputs, labels, _ = batch_data
            
            inputs = inputs.to(device, dtype=dtype)
            labels = labels.to(device)
            
            # Get covariance at SPDNet input
            if hasattr(model, 'backbone') or hasattr(model, 'feature_extractor'):
                # Model has a backbone
                if hasattr(model, 'backbone'):
                    features = model.backbone(inputs)
                else:
                    features = model.feature_extractor(inputs)
                
                if hasattr(model, 'cov_pooling'):
                    cov_input_batch = model.cov_pooling(features)
                else:
                    # Direct covariance computation
                    B, C, H, W = features.shape
                    features_reshaped = features.view(B, C, -1)
                    features_centered = features_reshaped - features_reshaped.mean(dim=2, keepdim=True)
                    cov_input_batch = torch.bmm(features_centered, features_centered.transpose(1, 2)) / (H * W)
            else:
                # Pure SPDNet - input is already covariance
                cov_input_batch = inputs
            
            # Forward pass to capture before LogEig
            matrices_before_logeig.clear()
            _ = model(inputs)
            
            if len(matrices_before_logeig) > 0:
                cov_before_logeig_batch = matrices_before_logeig[0]
            else:
                cov_before_logeig_batch = cov_input_batch.cpu()
            
            # Store by class and compute condition numbers
            for i in range(inputs.size(0)):
                class_idx = labels[i].item()
                
                if class_idx not in class_names:
                    continue
                
                # Check if we should store this sample
                should_store = analyze_all or samples_per_class[class_idx] < n_samples_per_class
                
                if should_store:
                    cov_input_matrix = cov_input_batch[i].cpu().numpy()
                    cov_output_matrix = cov_before_logeig_batch[i].cpu().numpy()
                    
                    # Compute condition numbers
                    eigvals_input = np.linalg.eigvalsh(cov_input_matrix)
                    eigvals_output = np.linalg.eigvalsh(cov_output_matrix)
                    
                    # Condition number = max eigenvalue / min eigenvalue
                    if eigvals_input.min() > 0:
                        cond_input = eigvals_input.max() / eigvals_input.min()
                        input_condition_numbers.append(cond_input)
                    
                    if eigvals_output.min() > 0:
                        cond_output = eigvals_output.max() / eigvals_output.min()
                        output_condition_numbers.append(cond_output)
                    
                    # Only store for visualization if not analyzing all
                    if not analyze_all:
                        cov_input[class_idx].append(cov_input_matrix)
                        cov_before_logeig[class_idx].append(cov_output_matrix)
                        samples_per_class[class_idx] += 1
            
            # Stop if we have enough samples (only when not analyzing all)
            if not analyze_all and all(count >= n_samples_per_class for count in samples_per_class.values()):
                break
    
    # Remove hook
    if hook_handle is not None:
        hook_handle.remove()
    
    # Compute conditioning statistics
    conditioning_stats = {}
    if len(input_condition_numbers) > 0:
        conditioning_stats['input_cond_mean'] = float(np.mean(input_condition_numbers))
        conditioning_stats['input_cond_std'] = float(np.std(input_condition_numbers))
    else:
        conditioning_stats['input_cond_mean'] = 0.0
        conditioning_stats['input_cond_std'] = 0.0
    
    if len(output_condition_numbers) > 0:
        conditioning_stats['output_cond_mean'] = float(np.mean(output_condition_numbers))
        conditioning_stats['output_cond_std'] = float(np.std(output_condition_numbers))
    else:
        conditioning_stats['output_cond_mean'] = 0.0
        conditioning_stats['output_cond_std'] = 0.0
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COVARIANCE MATRICES ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Conditioning statistics
    report_lines.append("CONDITIONING STATISTICS:")
    report_lines.append(f"  Input (after backbone/cov pooling):")
    report_lines.append(f"    Mean condition number: {conditioning_stats['input_cond_mean']:.2f}")
    report_lines.append(f"    Std condition number: {conditioning_stats['input_cond_std']:.2f}")
    report_lines.append(f"    Number of samples analyzed: {len(input_condition_numbers)}")
    report_lines.append(f"  Before LogEig (after SPDNet layers):")
    report_lines.append(f"    Mean condition number: {conditioning_stats['output_cond_mean']:.2f}")
    report_lines.append(f"    Std condition number: {conditioning_stats['output_cond_std']:.2f}")
    report_lines.append(f"    Number of samples analyzed: {len(output_condition_numbers)}")
    report_lines.append("")
    
    # Only generate detailed analysis if we saved matrices for visualization
    if not analyze_all:
        # Input dimensions
        sample_cov = list(cov_input.values())[0][0] if len(list(cov_input.values())[0]) > 0 else None
        if sample_cov is not None:
            report_lines.append(f"Input Covariance Matrix Dimension: {sample_cov.shape[0]}x{sample_cov.shape[1]}")
        
        sample_cov_out = list(cov_before_logeig.values())[0][0] if len(list(cov_before_logeig.values())[0]) > 0 else None
        if sample_cov_out is not None:
            report_lines.append(f"Before LogEig Matrix Dimension: {sample_cov_out.shape[0]}x{sample_cov_out.shape[1]}")
        report_lines.append("")
        
        # Width A4 = 21cm = 8.27 inches
        fig_width = 8.27
        fig_height = 6  # Fixed height for individual figures
        
        # Compute global quantiles (5% and 95%) for color scaling
        all_input_matrices = []
        all_output_matrices = []
        for class_idx in class_names.keys():
            if len(cov_input[class_idx]) > 0:
                all_input_matrices.extend(cov_input[class_idx])
            if len(cov_before_logeig[class_idx]) > 0:
                all_output_matrices.extend(cov_before_logeig[class_idx])
        
        if len(all_input_matrices) > 0:
            all_input_matrices = np.array(all_input_matrices)
            vmin_input = np.percentile(all_input_matrices, 5)
            vmax_input = np.percentile(all_input_matrices, 95)
        else:
            vmin_input, vmax_input = None, None
        
        if len(all_output_matrices) > 0:
            all_output_matrices = np.array(all_output_matrices)
            vmin_output = np.percentile(all_output_matrices, 5)
            vmax_output = np.percentile(all_output_matrices, 95)
        else:
            vmin_output, vmax_output = None, None
        
        # Generate individual figures for each class and sample
        for class_idx in sorted(class_names.keys()):
            class_name = class_names[class_idx]
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"CLASS: {class_name} (index: {class_idx})")
            report_lines.append(f"{'=' * 80}")
            
            # Input covariance analysis
            if len(cov_input[class_idx]) > 0:
                matrices_input = np.array(cov_input[class_idx])
                report_lines.append(f"\n[INPUT COVARIANCE - After Backbone/CovPooling]")
                report_lines.append(f"  Number of samples: {len(matrices_input)}")
                
                eigenvals_all = []
                spd_count = 0
                for mat in matrices_input:
                    eigvals = np.linalg.eigvalsh(mat)
                    eigenvals_all.extend(eigvals)
                    if np.min(eigvals) > 0:
                        spd_count += 1
                
                eigenvals_all = np.array(eigenvals_all)
                report_lines.append(f"  SPD matrices: {spd_count}/{len(matrices_input)} ({100*spd_count/len(matrices_input):.1f}%)")
                report_lines.append(f"  Matrix values - Mean: {matrices_input.mean():.6f}, Min: {matrices_input.min():.6f}, Max: {matrices_input.max():.6f}")
                report_lines.append(f"  Eigenvalues - Mean: {eigenvals_all.mean():.6f}, Min: {eigenvals_all.min():.6f}, Max: {eigenvals_all.max():.6f}")
                
                # Generate individual figures for input covariance
                for sample_idx, mat_input in enumerate(cov_input[class_idx]):
                    fig, (ax_mat, ax_eig) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
                    
                    # Covariance matrix
                    im = ax_mat.imshow(mat_input, cmap='gist_rainbow', aspect='equal', 
                                       vmin=vmin_input, vmax=vmax_input)
                    ax_mat.set_title(f'{class_name} - Sample {sample_idx}\nInput Covariance', fontsize=10)
                    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
                    
                    # Eigenvalues
                    eigvals = np.linalg.eigvalsh(mat_input)
                    eigvals_sorted = np.sort(eigvals)[::-1]
                    ax_eig.semilogy(eigvals_sorted, 'bo-', markersize=4, linewidth=1)
                    ax_eig.set_title(f'{class_name} - Sample {sample_idx}\nEigenvalues', fontsize=10)
                    ax_eig.set_xlabel('Index', fontsize=9)
                    ax_eig.set_ylabel('Eigenvalue (log)', fontsize=9)
                    ax_eig.grid(True, alpha=0.3)
                    ax_eig.set_aspect('auto')
                    
                    # Adjust subplot positions to match heights
                    plt.tight_layout()
                    pos_mat = ax_mat.get_position()
                    pos_eig = ax_eig.get_position()
                    ax_eig.set_position([pos_eig.x0, pos_mat.y0, pos_eig.width, pos_mat.height])
                    
                    # Save figure
                    if n_samples_per_class > 1:
                        filename = f"{class_name}_sample{sample_idx}.{save_format}"
                    else:
                        filename = f"{class_name}.{save_format}"
                    filepath = os.path.join(input_cov_dir, filename)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # Before LogEig analysis
            if len(cov_before_logeig[class_idx]) > 0:
                matrices_output = np.array(cov_before_logeig[class_idx])
                report_lines.append(f"\n[BEFORE LOG-EIG - After SPDNet layers]")
                report_lines.append(f"  Number of samples: {len(matrices_output)}")
                
                eigenvals_all = []
                spd_count = 0
                for mat in matrices_output:
                    eigvals = np.linalg.eigvalsh(mat)
                    eigenvals_all.extend(eigvals)
                    if np.min(eigvals) > 0:
                        spd_count += 1
                
                eigenvals_all = np.array(eigenvals_all)
                report_lines.append(f"  SPD matrices: {spd_count}/{len(matrices_output)} ({100*spd_count/len(matrices_output):.1f}%)")
                report_lines.append(f"  Matrix values - Mean: {matrices_output.mean():.6f}, Min: {matrices_output.min():.6f}, Max: {matrices_output.max():.6f}")
                report_lines.append(f"  Eigenvalues - Mean: {eigenvals_all.mean():.6f}, Min: {eigenvals_all.min():.6f}, Max: {eigenvals_all.max():.6f}")
                
                # Generate individual figures for before LogEig
                for sample_idx, mat_output in enumerate(cov_before_logeig[class_idx]):
                    fig, (ax_mat, ax_eig) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
                    
                    # Covariance matrix
                    im = ax_mat.imshow(mat_output, cmap='gist_rainbow', aspect='equal',
                                       vmin=vmin_output, vmax=vmax_output)
                    ax_mat.set_title(f'{class_name} - Sample {sample_idx}\nBefore LogEig', fontsize=10)
                    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
                    
                    # Eigenvalues
                    eigvals = np.linalg.eigvalsh(mat_output)
                    eigvals_sorted = np.sort(eigvals)[::-1]
                    ax_eig.semilogy(eigvals_sorted, 'ro-', markersize=4, linewidth=1)
                    ax_eig.set_title(f'{class_name} - Sample {sample_idx}\nEigenvalues', fontsize=10)
                    ax_eig.set_xlabel('Index', fontsize=9)
                    ax_eig.set_ylabel('Eigenvalue (log)', fontsize=9)
                    ax_eig.grid(True, alpha=0.3)
                    ax_eig.set_aspect('auto')
                    
                    # Adjust subplot positions to match heights
                    plt.tight_layout()
                    pos_mat = ax_mat.get_position()
                    pos_eig = ax_eig.get_position()
                    ax_eig.set_position([pos_eig.x0, pos_mat.y0, pos_eig.width, pos_mat.height])
                    
                    # Save figure
                    if n_samples_per_class > 1:
                        filename = f"{class_name}_sample{sample_idx}.{save_format}"
                    else:
                        filename = f"{class_name}.{save_format}"
                    filepath = os.path.join(output_cov_dir, filename)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"\nFigures saved:")
        report_lines.append(f"  Input covariance: {input_cov_dir}")
        report_lines.append(f"  Before LogEig: {output_cov_dir}")
    
    report_text = "\n".join(report_lines)
    
    return report_text, input_cov_dir, output_cov_dir, conditioning_stats
# ========================================
# FileName: utils.py
# Date: 29 juin 2023 - 17:14
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Utilities functions
# =========================================

import torch
from typing import Optional
import logging
from rich.logging import RichHandler
from typing import List, Tuple, Union, Callable, Dict
from math import prod

import matplotlib.pyplot as plt
import matplotlib


# ========================================
# Model checkpoint utilities
# ========================================
def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Load a model checkpoint with support for both full checkpoint and state_dict only.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to load the checkpoint into
    checkpoint_path : str
        Path to the checkpoint file
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into (if checkpoint contains optimizer state)
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to load state into (if checkpoint contains scheduler state)
    device : torch.device, optional
        Device to map the checkpoint to
    logger : logging.Logger, optional
        Logger for info messages
    
    Returns
    -------
    dict
        Checkpoint metadata (epoch, val_acc, val_loss, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    metadata = {}
    
    # Check if checkpoint is a dict with 'model_state_dict' key (full checkpoint)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch', None),
            'val_acc': checkpoint.get('val_acc', None),
            'val_loss': checkpoint.get('val_loss', None),
            'train_acc': checkpoint.get('train_acc', None),
            'train_loss': checkpoint.get('train_loss', None),
        }
        
        if logger:
            logger.info(f"Loaded checkpoint from epoch {metadata.get('epoch', 'unknown')}")
            if metadata.get('val_acc') is not None:
                logger.info(f"  - Val Acc: {metadata['val_acc']:.2f}%")
            if metadata.get('val_loss') is not None:
                logger.info(f"  - Val Loss: {metadata['val_loss']:.4f}")
    else:
        # If it's just a state_dict
        model.load_state_dict(checkpoint)
        if logger:
            logger.info("Loaded model state dict (no metadata)")
    
    return metadata


# ========================================
# Logging and formatting
# ========================================
def setup_matplotlib(use_tex: bool = False) -> None:
    """Setup Matplotlib according to my liking: black background with white
    test.

    Parameters
    ----------
    use_tex: bool
        Whether to render text using installed LaTeX
    """
    # Activate LaTeX rendering for matplotlib
    if use_tex:
        matplotlib.rcParams["text.usetex"] = True

    # Styling matplotlib
    plt.style.use("dark_background")
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.05
    matplotlib.rcParams["grid.linestyle"] = "dotted"
    matplotlib.rcParams["axes.facecolor"] = (0.1, 0.1, 0.1, 0.4)
    matplotlib.rcParams["font.family"] = "serif"
    # matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams["font.size"] = 11

def get_base_dataset(dataloader):
    """
    Helper function to get the base dataset from a DataLoader.
    Handles both direct datasets and Subset wrappers (including nested Subsets).
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        The underlying dataset (not wrapped in Subset)
    """
    dataset = dataloader.dataset
    # Recursively unwrap Subset layers until we reach the base dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def verify_dataset_labels(dataloader, num_classes, logger=None):
    """
    Verify that all labels in the dataset are within the valid range [0, num_classes-1].
    This function handles different dataset formats and label types.
    
    Args:
        dataloader: PyTorch DataLoader to verify
        num_classes: Expected number of classes
        logger: Optional logger for output (if None, uses print)
        
    Returns:
        tuple: (unique_labels_tensor, label_min, label_max)
        
    Raises:
        ValueError: If any label is outside the valid range
    """
    import torch
    
    log_fn = logger.info if logger else print
    
    all_labels = []
    
    # Check the format of dataset items by inspecting the first sample
    first_sample = dataloader.dataset[0]
    if len(first_sample) == 2:
        # Format: (data, label) - e.g., HyperLeaf
        for _, labels in dataloader.dataset:
            all_labels.append(labels)
    else:
        # Format: (data, label, metadata) - e.g., FUSAR, TenGeoP, S1SLC
        for _, labels, _ in dataloader.dataset:
            all_labels.append(labels)
    
    # Convert labels to a single tensor, handling different types
    if isinstance(all_labels[0], int):
        # Python integers
        all_labels = torch.tensor(all_labels)
    elif isinstance(all_labels[0], torch.Tensor):
        if all_labels[0].dim() == 0:
            # 0D tensors (scalars) - use stack
            all_labels = torch.stack(all_labels)
        else:
            # 1D or higher tensors - use cat
            all_labels = torch.cat(all_labels)
    else:
        # Fallback: try to convert to tensor
        all_labels = torch.tensor(all_labels)
    
    unique_labels = torch.unique(all_labels)
    label_min = unique_labels.min().item()
    label_max = unique_labels.max().item()
    
    log_fn(f"Unique labels in training data: {unique_labels.tolist()}")
    log_fn(f"Label range: [{label_min}, {label_max}]")
    
    if label_max >= num_classes:
        error_msg = (
            f"ERROR: Found label {label_max} but model only has {num_classes} classes! "
            f"Labels must be in range [0, {num_classes-1}]"
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    return unique_labels, label_min, label_max


class MatplotlibTrainingVisualizer:
    """Class to visualize training progress of one or several models."""

    def __init__(
        self,
        list_metrics: List[Dict],
        list_models: List[Dict],
        layout: Optional[Tuple[int, int]] = None,
        figure_options: Optional[Dict] = None,
        axes_options: Optional[Dict] = None,
    ) -> None:
        """Initialize the TrainingVisualizer object

        Parameters
        ----------
        list_metrics : List[Dict]
            List of dictionaries containing the metrics to plot for each model.
            Each dictionary should contain the following keys:
                - "name": Name of the metric

            And can contain the following keys:
                - scale: Scale of the metric (linear or log)
                - ylim: Limits of the y axis
                - ylabel: Label of the y axis

        list_models : List[Dict]
            List of dictionaries containing the models to plot. Each dictionary
            should contain the following keys:
                - "name": Name of the model

            And can contain the following keys:
                - plot_options: Options for the plot of the model that will be
                passed to the matplotlib plot function

        layout : Optional[Tuple[int, int]], optional
            Layout of the figure, by default None. Meaning that the figure will
            be a square with the number of subplots equal to the number of
            metrics.

        figure_options : Optional[Dict], optional
            Options for the figure, by default None

        axes_options : Optional[Dict], optional
            Options for the axes, by default None
        """

        assert len(list_metrics) > 0, "At least one metric must be provided"
        assert len(list_models) > 0, "At least one model must be provided"
        assert all([isinstance(metric, dict) for metric in list_metrics]), (
            "Metrics must be provided as dictionaries"
        )
        assert all([isinstance(model, dict) for model in list_models]), (
            "Models must be provided as dictionaries"
        )

        self.n_metrics = len(list_metrics)
        if layout is None:
            if not self.n_metrics == 1:
                self.layout = (1, 1)
            elif self.n_metrics == 2:
                self.layout = (2, 1)
            else:
                self.layout = (self.n_metrics // 2, 2)
        else:
            assert prod(layout) >= self.n_metrics, (
                f"Layout {layout} is not big enough to display {self.n_metrics} metrics"
            )
            self.layout = layout

        self.list_metrics = list_metrics
        self.list_models = list_models
        self.figure_options = figure_options
        self.axes_options = axes_options
        self.n_models = len(list_models)
        self.mapping_metric_axes = lambda i: (i // self.layout[0], i % self.layout[0])

        # Create the figure
        self.figure, self.axes = plt.subplots(*self.layout, **self.figure_options)
        if self.layout[1] == 1:
            self.axes = self.axes.reshape(1, self.n_metrics)
        elif self.layout[0] == 1:
            self.axes = self.axes.reshape(self.n_metrics, 1)

        # Setting axes options
        if self.axes_options is not None:
            for key, value in self.axes_options.items():
                setattr(self.axes, key, value)

        # Create the plots
        self.plots = []
        for i, metric in enumerate(self.list_metrics):
            self.plots.append([])
            _i, _j = self.mapping_metric_axes(i)
            for model in self.list_models:
                self.plots[i].append(
                    self.axes[_i, _j].plot(
                        [], [], label=model["name"], **model["plot_options"]
                    )[0]
                )

            # Setting metric options
            if "scale" in metric:
                self.axes[_i, _j].set_yscale(metric["scale"])
            if "ylim" in metric:
                self.axes[_i, _j].set_ylim(metric["ylim"])
            if "ylabel" in metric:
                self.axes[_i, _j].set_ylabel(metric["ylabel"])
            else:
                self.axes[_i, _j].set_ylabel(metric["name"])
            self.axes[_i, _j].set_xlabel("Epochs")

        # Initialize the data
        self.data = []
        for i, metric in enumerate(self.list_metrics):
            self.data.append([])
            for model in self.list_models:
                self.data[i].append([])

        # Initialize the legend
        self.legend = []
        for i, metric in enumerate(self.list_metrics):
            self.legend.append([])
            _i, _j = self.mapping_metric_axes(i)
            for j, model in enumerate(self.list_models):
                self.legend[i].append(self.axes[_i, _j].legend(loc="upper right"))

        # Initialize the epoch counter
        self.epoch = 0

    def __enter__(self):
        self.figure.canvas.draw()
        plt.show(block=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:  # An exception occurred inside the `with` block
            print(f"Exception: {exc_type}, {exc_value}, {traceback}")
        plt.close("all")

    def update(self, data: List[List[float]]) -> None:
        """Update the plots with the new data.

        Parameters
        ----------
        data : List[List[float]]
            List of data to update the plots with. The first dimension
            corresponds to the metrics, the second to the models.
        """

        assert len(data) == self.n_metrics, (
            f"Data must be a list of {self.n_metrics} elements"
        )
        assert all([len(data[i]) == self.n_models for i in range(self.n_metrics)]), (
            f"Each element of data must be a list of {self.n_models} elements"
        )

        # Update the data
        for i, _ in enumerate(self.list_metrics):
            for j, _ in enumerate(self.list_models):
                self.data[i][j].append(data[i][j])

        # Update the plots
        for i, _ in enumerate(self.list_metrics):
            _i, _j = self.mapping_metric_axes(i)
            for j, _ in enumerate(self.list_models):
                self.plots[i][j].set_data(list(range(self.epoch + 1)), self.data[i][j])
                self.axes[_i, _j].relim()
                self.axes[_i, _j].autoscale_view()

        self.epoch += 1

        # Update the figure
        plt.draw()
        plt.pause(0.05)


def setup_logging(
    level: str = "INFO", format: str = "%(message)s", markup: bool = True
) -> logging.Logger:
    """Setup logging to use rich handler

    Parameters
    ----------
    level : str, optional
        Logging level, by default "INFO"

    format : str, optional
        Format of the logging messages, by default "%(messages)"

    markup : bool, optional
        Whether to use rich markup, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    logging.basicConfig(
        level=level,
        format=format,
        datefmt="[%X]",
        handlers=[RichHandler(markup=markup, rich_tracebacks=True)],
    )
    return logging.getLogger("rich")


def format_params(params: dict, markup: bool = True) -> str:
    """Format a dictionary of parameters for logging
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    markup : bool, optional
        Whether to use rich markup, by default True
    Returns
    -------
    str
        Formatted string
    """
    string = ""
    for key, value in params.items():
        if markup:
            string += "    :black_circle_for_record: "
        else:
            string += "    - "
        string += f"{key}: {value}\n"
    return string


# ========================================
# SPD matrix generation
# TODO: Generate with Toepitz structure,
#       Generate with circulant structure
#       Generate with given eigenvalues distribution
#       Generate with Stiefel manifold
# ========================================
def generate_spd_matrices(
    dim: int,
    batch_size: Optional[int] = 1,
    device: Optional[str] = "cpu",
    seed: Optional[int] = 0,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    """
    Generate a batch of symmetric positive definite matrices

    Parameters
    ----------
    dim : int
        Dimension of the matrices

    batch_size : int, optional
        Batch size of the matrices, by default 1

    device : str, optional
        Device to put the matrices on, by default "cpu"

    seed : int, optional
        Seed for the random number generator, by default 0

    eps : float, optional
        Small value to add to the diagonal to ensure positive definiteness,
        by default 1e-6

    Returns
    -------
    torch.Tensor
        Batch of symmetric positive definite matrices
    """
    # Set the seed
    torch.manual_seed(seed)

    # Generate a batch of random matrices
    X = torch.randn(batch_size, dim, dim, device=device)

    # Compute the product of the matrices with their transposes
    X = torch.bmm(X, X.transpose(1, 2))

    # Add a small value to the diagonal to ensure positive definiteness
    X += eps * torch.eye(dim, device=device).expand(batch_size, dim, dim)

    if batch_size == 1:
        return X.squeeze(0)

    return X


# ========================================
# Functions to do checks
# ========================================
def is_spd(X: torch.Tensor, tol: Optional[float] = 1e-4) -> bool:
    """
    Check if a matrix is symmetric positive definite

    Parameters
    ----------
    X : torch.Tensor
        Matrix to check

    tol : float, optional
        Tolerance for the check, by default 1e-6

    Returns
    -------
    bool
        True if the matrix is symmetric positive definite, False otherwise
    """
    # Check if symmetric
    if not torch.allclose(X, X.t(), rtol=tol, atol=tol):
        return False

    # Check if positive definite
    try:
        torch.linalg.cholesky(X)
        return True
    except RuntimeError:
        distance = torch.linalg.norm(X - X.t())
        logging.debug("Matrix is not positive definite with tolerance {}".format(tol))
        logging.debug("Distance to symmetric matrix: {}".format(distance))
        return False


# ========================================
# Class Mismatch Handler
# ========================================
class ClassMismatchHandler:
    """
    Handles class mismatches between training and test datasets.
    
    Features:
    - Automatic detection of class mismatches
    - Filtering of invalid test samples
    - Proper loss computation with ignored classes
    - Detailed logging of class distribution differences
    """
    
    def __init__(self, train_n_classes: int, test_n_classes: int, train_label_names: Dict = None, test_label_names: Dict = None):
        """
        Initialize the class mismatch handler.
        
        Args:
            train_n_classes: Number of classes in training set
            test_n_classes: Number of classes in test set
            train_label_names: Optional mapping of class indices to names for training set
            test_label_names: Optional mapping of class indices to names for test set
        """
        self.train_n_classes = train_n_classes
        self.test_n_classes = test_n_classes
        self.train_label_names = train_label_names or {}
        self.test_label_names = test_label_names or {}
        
        self.has_mismatch = train_n_classes != test_n_classes
        self.valid_test_classes = set(range(min(train_n_classes, test_n_classes)))
        
        # Statistics
        self.total_test_samples = 0
        self.valid_test_samples = 0
        self.invalid_test_samples = 0
        self.skipped_batches = 0
        
        if self.has_mismatch:
            self._log_mismatch_info()
    
    def _log_mismatch_info(self):
        """Log detailed information about the class mismatch."""
        logger = logging.getLogger("rich")
        logger.warning("=" * 60)
        logger.warning("CLASS MISMATCH DETECTED")
        logger.warning("=" * 60)
        logger.warning(f"Training set classes: {self.train_n_classes}")
        logger.warning(f"Test set classes: {self.test_n_classes}")
        logger.warning(f"Valid test classes (intersection): {len(self.valid_test_classes)}")
        
        if self.train_n_classes < self.test_n_classes:
            invalid_classes = set(range(self.train_n_classes, self.test_n_classes))
            logger.warning(f"Test classes not in training: {list(invalid_classes)[:10]}...")
                
        logger.warning("Samples with invalid classes will be ignored during testing.")
        logger.warning("=" * 60)
    
    def filter_batch(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Filter a batch to only include samples with valid classes.
        
        Args:
            predictions: Model predictions [batch_size, n_classes]
            labels: True labels [batch_size]
            
        Returns:
            Tuple of (valid_predictions, valid_labels, valid_mask)
        """
        self.total_test_samples += len(labels)
        
        # Create mask for valid classes
        valid_mask = labels < self.train_n_classes
        
        valid_predictions = predictions[valid_mask]
        valid_labels = labels[valid_mask]
        
        self.valid_test_samples += valid_mask.sum().item()
        self.invalid_test_samples += (~valid_mask).sum().item()
        
        if valid_mask.sum() == 0:
            self.skipped_batches += 1
            
        return valid_predictions, valid_labels, valid_mask
    
    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Compute accuracy and other metrics for valid samples only.
        
        Args:
            predictions: Model predictions [batch_size, n_classes]
            labels: True labels [batch_size]
            
        Returns:
            Dictionary with computed metrics
        """
        valid_predictions, valid_labels, valid_mask = self.filter_batch(predictions, labels)
        
        if len(valid_predictions) == 0:
            return {"accuracy": 0.0, "valid_samples": 0}
        
        accuracy = (valid_predictions.argmax(dim=1) == valid_labels).float().mean() * 100
        
        return {
            "accuracy": accuracy.item(),
            "valid_samples": len(valid_labels),
            "total_samples": len(labels),
            "valid_ratio": len(valid_labels) / len(labels)
        }
    
    def get_statistics(self):
        """Get detailed statistics about the class mismatch handling."""
        return {
            "has_mismatch": self.has_mismatch,
            "train_classes": self.train_n_classes,
            "test_classes": self.test_n_classes,
            "valid_test_classes": len(self.valid_test_classes),
            "total_test_samples": self.total_test_samples,
            "valid_test_samples": self.valid_test_samples,
            "invalid_test_samples": self.invalid_test_samples,
            "skipped_batches": self.skipped_batches,
            "valid_sample_ratio": self.valid_test_samples / max(1, self.total_test_samples)
        }
    
    def log_final_statistics(self):
        """Log final statistics about class mismatch handling."""
        logger = logging.getLogger("rich")
        stats = self.get_statistics()
        
        if self.has_mismatch:
            logger.info("=" * 60)
            logger.info("CLASS MISMATCH HANDLING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total test samples processed: {stats['total_test_samples']}")
            logger.info(f"Valid test samples (used): {stats['valid_test_samples']}")
            logger.info(f"Invalid test samples (ignored): {stats['invalid_test_samples']}")
            logger.info(f"Valid sample ratio: {stats['valid_sample_ratio']:.2%}")
            logger.info(f"Batches skipped (all invalid): {stats['skipped_batches']}")
            
            if stats['valid_sample_ratio'] < 0.5:
                logger.warning("WARNING: Less than 50% of test samples are valid!")
                logger.warning("Consider using a different test set or class mapping strategy.")
            
            logger.info("=" * 60)
        else:
            logger.info("No class mismatch detected - all test samples are valid.")
    
    def prepare_confusion_matrix_data(self, all_predictions, all_labels):
        """
        Prepare data for confusion matrix, handling class mismatches.
        
        Args:
            all_predictions: List of all predictions
            all_labels: List of all true labels
            
        Returns:
            Tuple of (filtered_predictions, filtered_labels, label_names)
        """
        # Filter to only include valid classes
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(all_predictions, all_labels):
            if label < self.train_n_classes:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        # Get label names for the classes that appear in filtered data
        unique_labels = sorted(set(filtered_labels))
        label_names = []
        
        for label in unique_labels:
            if label in self.train_label_names:
                label_names.append(self.train_label_names[label])
            else:
                label_names.append(f"class_{label}")
        
        return filtered_predictions, filtered_labels, label_names
    
    def create_loss_function(self, base_loss_fn=None):
        """
        Create a loss function that handles class mismatches.
        
        Args:
            base_loss_fn: Base loss function (default: CrossEntropyLoss)
            
        Returns:
            Loss function that ignores invalid classes
        """
        import torch.nn as nn
        if base_loss_fn is None:
            return nn.CrossEntropyLoss(ignore_index=-1)
        return base_loss_fn


def log_dataset_comparison(train_dataset, test_dataset):
    """
    Log detailed comparison between training and test datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset  
    """
    logger = logging.getLogger("rich")
    logger.info("=" * 80)
    logger.info("DATASET COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Training dataset:")
    logger.info(f"  - Samples: {len(train_dataset)}")
    logger.info(f"  - Classes: {train_dataset.n_classes}")
    logger.info(f"  - Matrix dimension: {train_dataset.d}x{train_dataset.d}")
    
    logger.info(f"Test dataset:")
    logger.info(f"  - Samples: {len(test_dataset)}")
    logger.info(f"  - Classes: {test_dataset.n_classes}")
    logger.info(f"  - Matrix dimension: {test_dataset.d}x{test_dataset.d}")
    
    # Check for potential data leakage indicators
    if hasattr(train_dataset, 'metadata_sets') and hasattr(test_dataset, 'metadata_sets'):
        train_sets = getattr(train_dataset, 'metadata_sets', 'unknown')
        test_sets = getattr(test_dataset, 'metadata_sets', 'unknown')
        logger.info(f"Training uses metadata sets: {train_sets}")
        logger.info(f"Test uses metadata sets: {test_sets}")
        
        if train_sets == test_sets:
            logger.warning("⚠️  POTENTIAL DATA LEAKAGE: Both datasets use the same metadata sets!")
            logger.warning("⚠️  Consider using different sets (e.g., Set 1 for train, Set 2 for test)")
    
    logger.info("=" * 80)

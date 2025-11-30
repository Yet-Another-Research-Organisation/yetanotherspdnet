"""
Shared training functions for experiments.

Contains common training, validation, and testing functions used across
different model architectures trained on GPR datasets.
################## TO DO ##################
- Add more metric for multi-class classification
"""

import os
import time
import torch
import numpy as np
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

from .rich_logging import setup_logging
from .metrics import MetricsWriter
from sklearn.metrics import confusion_matrix


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, dtype=torch.float32):
    """
    Train model for one epoch with progress tracking.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number
    dtype : torch.dtype, default=torch.float32
        Data type for model inputs

    Returns
    -------
    tuple
        (epoch_loss, epoch_acc) - Training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        TextColumn("•"),
        TextColumn("Acc: {task.fields[acc]:.2f}%"),
        console=console,
    ) as progress:
        train_task = progress.add_task(
            f"[green]Epoch {epoch}", total=len(train_loader), loss=0.0, acc=0.0
        )

        for i, elements in enumerate(train_loader):
            size_elmts = len(elements)
            if size_elmts == 2:
                inputs, labels = elements
            elif size_elmts == 3:
                inputs, labels, metadata = elements
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            batch_loss = running_loss / (i + 1)
            batch_acc = 100.0 * correct / total
            progress.update(train_task, advance=1, loss=batch_loss, acc=batch_acc)

    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    # Print summary after the progress bar
    console.print(
        f"[bold green]Epoch {epoch} Summary:[/bold green] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%"
    )

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, dtype=torch.float32):
    """
    Validate model with progress tracking.

    Parameters
    ----------
    model : torch.nn.Module
        The model to validate
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to validate on
    dtype : torch.dtype, default=torch.float32
        Data type for model inputs

    Returns
    -------
    tuple
        (val_loss, val_acc) - Validation loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        console=console,
    ) as progress:
        val_task = progress.add_task(
            "[cyan]Validation", total=len(val_loader), loss=0.0
        )

        with torch.no_grad():
            for i, elements in enumerate(val_loader):
                size_elmts = len(elements)
                if size_elmts == 2:
                    inputs, labels = elements
                elif size_elmts == 3:
                    inputs, labels, metadata = elements
                inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                current_loss = val_loss / (i + 1)
                progress.update(val_task, advance=1, loss=current_loss)

    # Calculate validation statistics
    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    console.print(
        f"[bold cyan]Validation Results:[/bold cyan] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
    )

    return val_loss, val_acc


def evaluate_model(
    model, test_loader, criterion, device, class_names, logger=setup_logging(), dtype=torch.float32
):
    """
    Test model and generate comprehensive evaluation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The model to test
    test_loader : torch.utils.data.DataLoader
        Test data loader
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to test on
    class_names : list
        List of class names
    logger: logging.Logger
        logger for the evaluation
    dtype : torch.dtype, default=torch.float32
        Data type for model inputs

    Returns
    -------
    tuple
        (test_loss, test_acc) - Test loss and accuracy
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # For confusion matrix and metrics
    all_preds = []
    all_targets = []

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        TextColumn("•"),
        TextColumn("Acc: {task.fields[acc]:.2f}%"),
        console=console,
    ) as progress:
        test_task = progress.add_task(
            "[magenta]Testing", total=len(test_loader), loss=0.0, acc=0.0
        )

        with torch.no_grad():
            for i, elements in enumerate(test_loader):
                size_elmts = len(elements)
                if size_elmts == 2:
                    inputs, labels = elements
                elif size_elmts == 3:
                    inputs, labels, metadata = elements
                inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                current_correct = predicted.eq(labels).sum().item()
                correct += current_correct

                # Store predictions and targets for later analysis
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                # Update progress bar
                current_loss = test_loss / (i + 1)
                current_acc = 100.0 * correct / total
                progress.update(
                    test_task, advance=1, loss=current_loss, acc=current_acc
                )

    # Calculate test statistics
    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    console.print(
        f"[bold magenta]Test Results:[/bold magenta] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%"
    )

    # Debug class information
    unique_targets = np.unique(all_targets)
    unique_preds = np.unique(all_preds)
    logger.info("Debug info:")
    logger.info(f"  Total class names: {len(class_names)}")
    logger.info(f"  Class names: {class_names}")
    logger.info(
        f"  Unique targets in test set: {len(unique_targets)} -> {unique_targets}"
    )
    logger.info(f"  Unique predictions: {len(unique_preds)} -> {unique_preds}")
    logger.info(f"  Target range: {min(all_targets)} to {max(all_targets)}")
    logger.info(f"  Prediction range: {min(all_preds)} to {max(all_preds)}")

    # Check for class mismatch
    if len(unique_preds) != len(class_names):
        logger.warning("Class mismatch detected!")
        logger.warning(
            f"  Expected {len(class_names)} classes, but got {len(unique_preds)} unique predictions"
        )

        # Find missing classes in predictions
        all_class_indices = set(range(len(class_names)))
        missing_in_preds = all_class_indices - set(unique_preds)
        if missing_in_preds:
            logger.warning(
                f"  Classes missing in predictions: {sorted(missing_in_preds)}"
            )
            for missing_idx in sorted(missing_in_preds):
                if missing_idx < len(class_names):
                    logger.warning(f"    {missing_idx}: {class_names[missing_idx]}")

    # Print classification report with only the classes present in the data
    # Filter class_names to only include classes that appear in targets
    unique_targets_set = set(unique_targets)
    filtered_class_names = [str(class_names[i]) for i in sorted(unique_targets_set) if i < len(class_names)]
    filtered_labels = sorted(unique_targets_set)
    
    report = classification_report(
        all_targets, all_preds, labels=filtered_labels, target_names=filtered_class_names, zero_division=0
    )

    logger.info("Classification Report:")
    logger.info(report)

    # Calculate metrics
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    f1 = f1_score(all_targets, all_preds, average="weighted")

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=filtered_labels)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    report_dict = classification_report(
        all_targets, all_preds, labels=filtered_labels, 
        target_names=filtered_class_names, zero_division=0, output_dict=True
    )
    
    for i, class_name in enumerate(filtered_class_names):
        if class_name in report_dict:
            per_class_metrics[class_name] = {
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1-score": report_dict[class_name]["f1-score"],
                "support": report_dict[class_name]["support"],
            }

    return test_loss, test_acc, conf_matrix, report, precision, recall, f1, per_class_metrics


def setup_training_environment(args, experiment_name=None):
    """
    Setup training environment including device, seeding, and metrics writer.

    Parameters
    ----------
    args : object
        Arguments object containing configuration
    experiment_name : str, optional
        Name for the experiment. If None, generates one based on timestamp

    Returns
    -------
    tuple
        (device, generator, metrics_writer, logger)
    """
    # Setup seed
    generator = torch.Generator()
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        generator.manual_seed(int(args.seed))

    # Initialize metrics writer for dashboard
    if experiment_name is None:
        experiment_name = f"training_{int(time.time())}"

    metrics_writer = MetricsWriter(
        log_dir=args.storage_path, experiment_name=experiment_name
    )

    logger = setup_logging(file=os.path.join(args.storage_path, "log.txt"))

    return generator, metrics_writer, logger


def create_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    metrics_writer,
    logger=setup_logging(),
    dtype=torch.float32,
):
    """
    Execute the main training loop with optional validation, early stopping and model saving.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader or None
        Validation data loader. If None, validation is skipped.
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    device : torch.device
        Device to train on
    args : object
        Arguments object containing training configuration
    metrics_writer : MetricsWriter
        Metrics writer for dashboard logging
    logger: logging.Logger
        logger for the evaluation

    Returns
    -------
    tuple
        (best_val_acc, best_model_path, training_history)
    """
    console = Console()
    
    # Check if validation is enabled
    use_validation = val_loader is not None
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_path = None
    if args.save_best_model:
        best_model_path = os.path.join(args.storage_path, "best_model.pth")
        os.makedirs(args.storage_path, exist_ok=True)

    # Early stopping (disabled when no validation)
    early_stopping_counter = 0
    use_early_stopping = use_validation and args.early_stopping_patience > 0

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []

    # Start training for dashboard
    metrics_writer.start_training(total_epochs=args.epochs)

    for epoch in range(1, args.epochs + 1):
        console.rule(f"[bold]Epoch {epoch}/{args.epochs}[/bold]")

        # Update model epoch (for batchnorm_start_epoch feature)
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
            if hasattr(model, 'batchnorm_start_epoch') and model.batchnorm_start_epoch > 0:
                if epoch == model.batchnorm_start_epoch:
                    logger.info(f"🔄 Epoch {epoch}: Activating BatchNorm layers (replacing Identity layers)")
                    logger.info(f"   → Model architecture changed: BatchNormSPDMean layers are now active")
                    logger.info(f"   → Training behavior will differ from epochs 1-{model.batchnorm_start_epoch-1}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.6f}")
        learning_rates.append(current_lr)

        # Train and validate
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, dtype=dtype
        )
        
        if use_validation:
            val_loss, val_acc = validate(model, val_loader, criterion, device, dtype=dtype)
        else:
            # No validation - use training metrics for tracking
            val_loss, val_acc = train_loss, train_acc

        # Extract learnable epsilon values from ReEigLearnable layers
        reeig_eps_values = {}
        reeig_learnable_count = 0
        reeig_standard_count = 0
        
        for name, module in model.named_modules():
            module_class = module.__class__.__name__
            if module_class == 'ReEigLearnable':
                reeig_learnable_count += 1
                # Get the current eps value (it's a Parameter)
                eps_value = module.eps.item() if hasattr(module.eps, 'item') else module.eps
                # Clean up the name to make it more readable
                clean_name = name.replace('spdnet_layers.', 'layer_').replace('spdnet.spdnet_layers.', 'layer_')
                reeig_eps_values[clean_name] = float(eps_value)
            elif module_class == 'ReEig':
                reeig_standard_count += 1
        
        # Log debug info on first epoch
        if epoch == 1:
            logger.info(f"Model contains {reeig_learnable_count} ReEigLearnable layers and {reeig_standard_count} standard ReEig layers")
            if reeig_learnable_count > 0:
                logger.info(f"Learnable epsilon values will be tracked: {list(reeig_eps_values.keys())}")
        
        # Log metrics for dashboard
        metrics_writer.log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc / 100.0,  # Convert from percentage to decimal
            val_loss=val_loss,
            val_acc=val_acc / 100.0,  # Convert from percentage to decimal
            learning_rate=current_lr,
            reeig_eps_values=reeig_eps_values if reeig_eps_values else None,
        )

        # Store training history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update scheduler
        if scheduler is not None and hasattr(scheduler, "step"):
            # For schedulers that need metrics (like ReduceLROnPlateau)
            if (
                "warmup_plateau" in str(type(scheduler).__name__).lower()
                or "plateau" in str(type(scheduler).__name__).lower()
            ):
                if use_validation:
                    scheduler.step(val_loss)
                else:
                    # Skip plateau-based schedulers when no validation
                    logger.info("Skipping plateau-based scheduler step (no validation)")
            else:
                scheduler.step()

        # Save best model (only when using validation)
        if use_validation and val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            # Update best metrics for dashboard
            metrics_writer.update_status(
                {"best_val_acc": val_acc / 100.0, "best_val_loss": val_loss}
            )
            if args.save_best_model:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                }
                if scheduler is not None:
                    save_dict["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(save_dict, best_model_path)
                logger.info(
                    f"Best model saved with validation accuracy: {val_acc:.2f}%"
                )
        elif use_validation:
            early_stopping_counter += 1

        # Early stopping check (only when using validation)
        if use_early_stopping and early_stopping_counter >= args.early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs (patience: {args.early_stopping_patience})"
            )
            metrics_writer.update_status({"status": "early_stopped"})
            break

        # Log progress
        if use_validation:
            logger.info(
                f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            if early_stopping_counter > 0:
                logger.info(
                    f"Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}"
                )
        else:
            logger.info(
                f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% (no validation)"
            )

    # Save final model when no validation is used
    if not use_validation and args.save_best_model:
        final_model_path = os.path.join(args.storage_path, "final_model.pth")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "train_loss": train_loss,
        }
        if scheduler is not None:
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(save_dict, final_model_path)
        logger.info(f"Final model saved with training accuracy: {train_acc:.2f}%")
        best_model_path = final_model_path
        best_val_acc = train_acc  # Use training accuracy as "best" metric

    # Mark training as completed for dashboard  
    metrics_writer.update_status({"status": "completed"})

    # Training history
    training_history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "learning_rates": learning_rates,
        "final_epoch": epoch,
    }

    return best_val_acc, best_model_path, training_history

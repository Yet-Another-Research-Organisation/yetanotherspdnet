"""
Custom learning rate schedulers with warmup functionality.

This module provides various learning rate schedulers that combine warmup phases
with different decay strategies commonly used in deep learning.
"""

import math
import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
)
import inspect


class WarmupReduceLROnPlateau:
    """
    Learning rate scheduler that combines warmup with ReduceLROnPlateau.

    During warmup phase, learning rate increases from initial LR to target LR
    using the specified warmup strategy. After warmup, ReduceLROnPlateau is used.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    warmup_epochs : int
        Number of warmup epochs
    target_lr : float, optional
        Target learning rate after warmup. If None, uses initial LR from optimizer
    warmup_type : str, default='linear'
        Type of warmup: 'linear', 'cosine', 'exponential', 'polynomial'
    warmup_power : float, default=2.0
        Power for polynomial warmup (only used when warmup_type='polynomial')
    mode : str, default='min'
        Mode for ReduceLROnPlateau
    factor : float, default=0.5
        Factor for ReduceLROnPlateau
    patience : int, default=10
        Patience for ReduceLROnPlateau
    **plateau_kwargs
        Additional kwargs for ReduceLROnPlateau
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        target_lr=None,
        warmup_type="linear",
        warmup_power=2.0,
        mode="min",
        factor=0.5,
        patience=10,
        **plateau_kwargs,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_power = warmup_power
        self.current_epoch = 0

        # Store initial learning rates
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set target learning rates
        if target_lr is not None:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        else:
            self.target_lrs = self.initial_lrs.copy()

        # Create ReduceLROnPlateau scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, **plateau_kwargs
        )

        # Initialize with starting LR if warmup is used
        if warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lrs[0]

    def _get_warmup_lr(self, epoch):
        """Calculate learning rate for warmup phase."""
        if self.warmup_epochs == 0:
            return self.target_lrs

        progress = epoch / self.warmup_epochs

        if self.warmup_type == "linear":
            warmup_factor = progress
        elif self.warmup_type == "cosine":
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.warmup_type == "exponential":
            warmup_factor = progress**2
        elif self.warmup_type == "polynomial":
            warmup_factor = progress**self.warmup_power
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return [
            initial_lr + (target_lr - initial_lr) * warmup_factor
            for initial_lr, target_lr in zip(self.initial_lrs, self.target_lrs)
        ]

    def step(self, metrics=None):
        """Step the scheduler."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lrs = self._get_warmup_lr(self.current_epoch)
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group["lr"] = lr
        else:
            # Plateau phase
            if metrics is not None:
                self.plateau_scheduler.step(metrics)

    def state_dict(self):
        """Return state of the scheduler."""
        return {
            "current_epoch": self.current_epoch,
            "initial_lrs": self.initial_lrs,
            "target_lrs": self.target_lrs,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.initial_lrs = state_dict["initial_lrs"]
        self.target_lrs = state_dict["target_lrs"]
        self.plateau_scheduler.load_state_dict(state_dict["plateau_scheduler"])


class WarmupCosineAnnealingLR:
    """
    Learning rate scheduler that combines warmup with cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    warmup_epochs : int
        Number of warmup epochs
    T_max : int
        Maximum number of epochs for cosine annealing
    target_lr : float, optional
        Target learning rate after warmup
    warmup_type : str, default='linear'
        Type of warmup: 'linear', 'cosine', 'exponential', 'polynomial'
    warmup_power : float, default=2.0
        Power for polynomial warmup
    eta_min : float, default=0
        Minimum learning rate for cosine annealing
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_max,
        target_lr=None,
        warmup_type="linear",
        warmup_power=2.0,
        eta_min=0,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_power = warmup_power
        self.current_epoch = 0

        # Store initial learning rates
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set target learning rates
        if target_lr is not None:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        else:
            self.target_lrs = self.initial_lrs.copy()

        # Create cosine annealing scheduler
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

        # Set target LR for cosine scheduler
        for param_group, target_lr in zip(optimizer.param_groups, self.target_lrs):
            param_group["lr"] = target_lr

        # Initialize with starting LR if warmup is used
        if warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lrs[0]

    def _get_warmup_lr(self, epoch):
        """Calculate learning rate for warmup phase."""
        if self.warmup_epochs == 0:
            return self.target_lrs

        progress = epoch / self.warmup_epochs

        if self.warmup_type == "linear":
            warmup_factor = progress
        elif self.warmup_type == "cosine":
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.warmup_type == "exponential":
            warmup_factor = progress**2
        elif self.warmup_type == "polynomial":
            warmup_factor = progress**self.warmup_power
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return [
            initial_lr + (target_lr - initial_lr) * warmup_factor
            for initial_lr, target_lr in zip(self.initial_lrs, self.target_lrs)
        ]

    def step(self):
        """Step the scheduler."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lrs = self._get_warmup_lr(self.current_epoch)
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group["lr"] = lr
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step()

    def state_dict(self):
        """Return state of the scheduler."""
        return {
            "current_epoch": self.current_epoch,
            "initial_lrs": self.initial_lrs,
            "target_lrs": self.target_lrs,
            "cosine_scheduler": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.initial_lrs = state_dict["initial_lrs"]
        self.target_lrs = state_dict["target_lrs"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler"])


class WarmupStepLR:
    """
    Learning rate scheduler that combines warmup with step decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    warmup_epochs : int
        Number of warmup epochs
    step_size : int
        Step size for step decay
    target_lr : float, optional
        Target learning rate after warmup
    warmup_type : str, default='linear'
        Type of warmup: 'linear', 'cosine', 'exponential', 'polynomial'
    warmup_power : float, default=2.0
        Power for polynomial warmup
    gamma : float, default=0.1
        Multiplicative factor for step decay
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        step_size,
        target_lr=None,
        warmup_type="linear",
        warmup_power=2.0,
        gamma=0.1,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_power = warmup_power
        self.current_epoch = 0

        # Store initial learning rates
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set target learning rates
        if target_lr is not None:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        else:
            self.target_lrs = self.initial_lrs.copy()

        # Create step scheduler
        self.step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Set target LR for step scheduler
        for param_group, target_lr in zip(optimizer.param_groups, self.target_lrs):
            param_group["lr"] = target_lr

        # Initialize with starting LR if warmup is used
        if warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lrs[0]

    def _get_warmup_lr(self, epoch):
        """Calculate learning rate for warmup phase."""
        if self.warmup_epochs == 0:
            return self.target_lrs

        progress = epoch / self.warmup_epochs

        if self.warmup_type == "linear":
            warmup_factor = progress
        elif self.warmup_type == "cosine":
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.warmup_type == "exponential":
            warmup_factor = progress**2
        elif self.warmup_type == "polynomial":
            warmup_factor = progress**self.warmup_power
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return [
            initial_lr + (target_lr - initial_lr) * warmup_factor
            for initial_lr, target_lr in zip(self.initial_lrs, self.target_lrs)
        ]

    def step(self):
        """Step the scheduler."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lrs = self._get_warmup_lr(self.current_epoch)
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group["lr"] = lr
        else:
            # Step decay phase
            self.step_scheduler.step()

    def state_dict(self):
        """Return state of the scheduler."""
        return {
            "current_epoch": self.current_epoch,
            "initial_lrs": self.initial_lrs,
            "target_lrs": self.target_lrs,
            "step_scheduler": self.step_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.initial_lrs = state_dict["initial_lrs"]
        self.target_lrs = state_dict["target_lrs"]
        self.step_scheduler.load_state_dict(state_dict["step_scheduler"])


class WarmupExponentialLR:
    """
    Learning rate scheduler that combines warmup with exponential decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    warmup_epochs : int
        Number of warmup epochs
    target_lr : float, optional
        Target learning rate after warmup
    warmup_type : str, default='linear'
        Type of warmup: 'linear', 'cosine', 'exponential', 'polynomial'
    warmup_power : float, default=2.0
        Power for polynomial warmup
    gamma : float, default=0.95
        Multiplicative factor for exponential decay
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        target_lr=None,
        warmup_type="linear",
        warmup_power=2.0,
        gamma=0.95,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_power = warmup_power
        self.current_epoch = 0

        # Store initial learning rates
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set target learning rates
        if target_lr is not None:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        else:
            self.target_lrs = self.initial_lrs.copy()

        # Create exponential scheduler
        self.exp_scheduler = ExponentialLR(optimizer, gamma=gamma)

        # Set target LR for exponential scheduler
        for param_group, target_lr in zip(optimizer.param_groups, self.target_lrs):
            param_group["lr"] = target_lr

        # Initialize with starting LR if warmup is used
        if warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lrs[0]

    def _get_warmup_lr(self, epoch):
        """Calculate learning rate for warmup phase."""
        if self.warmup_epochs == 0:
            return self.target_lrs

        progress = epoch / self.warmup_epochs

        if self.warmup_type == "linear":
            warmup_factor = progress
        elif self.warmup_type == "cosine":
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.warmup_type == "exponential":
            warmup_factor = progress**2
        elif self.warmup_type == "polynomial":
            warmup_factor = progress**self.warmup_power
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return [
            initial_lr + (target_lr - initial_lr) * warmup_factor
            for initial_lr, target_lr in zip(self.initial_lrs, self.target_lrs)
        ]

    def step(self):
        """Step the scheduler."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lrs = self._get_warmup_lr(self.current_epoch)
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group["lr"] = lr
        else:
            # Exponential decay phase
            self.exp_scheduler.step()

    def state_dict(self):
        """Return state of the scheduler."""
        return {
            "current_epoch": self.current_epoch,
            "initial_lrs": self.initial_lrs,
            "target_lrs": self.target_lrs,
            "exp_scheduler": self.exp_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.initial_lrs = state_dict["initial_lrs"]
        self.target_lrs = state_dict["target_lrs"]
        self.exp_scheduler.load_state_dict(state_dict["exp_scheduler"])


def get_valid_kwargs(func, kwargs):
    """Filter kwargs to only include parameters that the function accepts"""
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}


def create_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Factory function to create schedulers by name.

    Parameters
    ----------
    scheduler_name : str or None
        Name of the scheduler to create. If None, no scheduler is used (fixed learning rate).
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    **kwargs
        Scheduler-specific arguments

    Returns
    -------
    scheduler or None
        The created scheduler instance, or None for fixed learning rate

    Raises
    ------
    ValueError
        If scheduler_name is not supported
    """
    
    # Handle no scheduler case (fixed learning rate)
    if scheduler_name is None:
        return None

    schedulers = {
        "warmup_plateau": WarmupReduceLROnPlateau,
        "warmup_cosine": WarmupCosineAnnealingLR,
        "warmup_step": WarmupStepLR,
        "warmup_exponential": WarmupExponentialLR,
        "plateau": lambda opt, **kw: ReduceLROnPlateau(
            opt, **get_valid_kwargs(ReduceLROnPlateau, kw)
        ),
        "cosine": lambda opt, **kw: CosineAnnealingLR(
            opt, **get_valid_kwargs(CosineAnnealingLR, kw)
        ),
        "step": lambda opt, **kw: StepLR(opt, **get_valid_kwargs(StepLR, kw)),
        "exponential": lambda opt, **kw: ExponentialLR(
            opt, **get_valid_kwargs(ExponentialLR, kw)
        ),
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available schedulers: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name](optimizer, **kwargs)

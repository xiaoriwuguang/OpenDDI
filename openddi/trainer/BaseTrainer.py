import copy
import time
import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseTrainer(ABC):
    """
    Base Trainer class that provides common functionality for training neural networks.
    Supports both multiclass and multilabel classification tasks.

    Features:
    - Device management (CPU/GPU)
    - Mixed precision training with GradScaler
    - Memory tracking
    - Time tracking
    - Common training loop structure
    - Result logging and saving
    """

    def __init__(self, args, logger, dataset, model, optimizer):
        """
        Initialize the BaseTrainer.

        Args:
            args: Configuration arguments
            logger: Logger instance for logging
            dataset: Dataset object containing data loaders
            model: Neural network model to train
            optimizer: Optimizer for training
        """
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        # Initialize tracking variables
        self.time0 = time.time()
        self.device = self._setup_device()

        # Enable TF32 acceleration for A100 GPUs
        self._setup_tf32()

    def _setup_device(self) -> torch.device:
        """
        Setup and return the appropriate device for training.

        Returns:
            torch.device: Device to use for training
        """
        device_name = getattr(self.args, 'device', None)
        if device_name:
            return torch.device(device_name)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _setup_tf32(self):
        """Enable TF32 acceleration for A100 GPUs if available."""
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _setup_scaler(self) -> torch.cuda.amp.GradScaler:
        """
        Setup GradScaler for mixed precision training.

        Returns:
            torch.cuda.amp.GradScaler: Configured scaler
        """
        return torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

    def _move_data_to_device(self):
        """Move dataset data object to the training device."""
        if hasattr(self.dataset, 'data_o'):
            self.dataset.data_o = self.dataset.data_o.to(self.device)

    def _get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in MB.

        Returns:
            float: Memory usage in megabytes
        """
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time since training started.

        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.time0

    def _log_training_progress(self, epoch: int, train_metrics: Dict[str, float],
                             val_metrics: Dict[str, float]):
        """
        Log training progress for current epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        train_str = " | ".join([f"Train {k}={v:.4f}" for k, v in train_metrics.items()])
        val_str = " | ".join([f"Val {k}={v:.4f}" for k, v in val_metrics.items()])

        print(f"Epoch {epoch+1:02d}: {train_str} | {val_str}")
        print(f"Memory: {self._get_memory_usage():.2f} MB | Time: {self._get_elapsed_time():.3f}s")

    def _save_results(self, test_metrics: Dict[str, float], model_name: str = "Model"):
        """
        Save test results to output file.

        Args:
            test_metrics: Dictionary of test metrics
            model_name: Name of the model for logging
        """
        out_path = getattr(self.args, 'out_file', 'result.txt')
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        metrics_str = "   ".join([f"{v:.4f}" for v in test_metrics.values()])
        time_str = f"{self._get_elapsed_time():.3f}"
        memory_str = f"{self._get_memory_usage():.2f}"

        with open(out_path, "a") as f:
            f.write(f"{model_name}  {metrics_str}   time:{time_str}   memory:{memory_str}\n")

    def _prepare_batch_data(self, batch_data, task_type: str = 'multiclass') -> torch.Tensor:
        """
        Prepare and convert batch data to appropriate tensor format.

        Args:
            batch_data: Batch data from data loader
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            torch.Tensor: Prepared labels tensor
        """
        labels = batch_data[2]

        if task_type == 'multiclass':
            return torch.as_tensor(np.array(labels), dtype=torch.long, device=self.device)
        elif task_type == 'multilabel':
            return torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @abstractmethod
    def _get_loss_function(self, task_type: str):
        """
        Get the appropriate loss function for the task.

        Args:
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Loss function
        """
        pass

    @abstractmethod
    def _compute_metrics(self, y_true: np.ndarray, y_logits: np.ndarray,
                        task_type: str) -> Dict[str, float]:
        """
        Compute evaluation metrics for predictions.

        Args:
            y_true: Ground truth labels
            y_logits: Model logits
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Dictionary of computed metrics
        """
        pass

    @abstractmethod
    def _train_epoch(self, epoch: int, loss_fct, scaler: torch.cuda.amp.GradScaler,
                    task_type: str) -> Tuple[Dict[str, float], float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            loss_fct: Loss function
            scaler: GradScaler for mixed precision
            task_type: Type of task

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        pass

    @abstractmethod
    def _evaluate(self, loader, loss_fct, task_type: str) -> Tuple[Dict[str, float], float]:
        """
        Evaluate the model on given data loader.

        Args:
            loader: Data loader for evaluation
            loss_fct: Loss function
            task_type: Type of task

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        pass

    def train(self):
        """
        Main training method. Determines task type and starts appropriate training.
        """
        # Move data to device
        self._move_data_to_device()

        # Determine task type based on matrix argument
        if getattr(self.args, 'matrix', None) in ['multilabel', 'twosides']:
            self._train_multilabel()
        else:
            self._train_multiclass()

    def _train_multiclass(self):
        """Training loop for multiclass classification."""
        print('Start Training (multiclass)...')

        # Setup training components
        loss_fct = self._get_loss_function('multiclass')
        scaler = self._setup_scaler()

        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, loss_fct, scaler, 'multiclass')

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, loss_fct, 'multiclass')

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, loss_fct, 'multiclass')

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[Model] Test {metrics_str}")
        self._save_results(test_metrics, "Model")

    def _train_multilabel(self):
        """Training loop for multilabel classification."""
        print('Start Training (multilabel)...')

        # Setup training components
        loss_fct = self._get_loss_function('multilabel')
        scaler = self._setup_scaler()

        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, loss_fct, scaler, 'multilabel')

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, loss_fct, 'multilabel')

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, loss_fct, 'multilabel')

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[Model] Test {metrics_str}")
        self._save_results(test_metrics, "Model")

    # Legacy method names for backward compatibility
    def train_binary(self):
        """Legacy method - redirects to main train method."""
        self.train()

    def train_multi(self):
        """Legacy method - redirects to main train method."""
        self.train()
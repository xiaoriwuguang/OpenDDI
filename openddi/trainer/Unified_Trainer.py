import copy
import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc
)
from typing import Dict, Any, Tuple, Optional
from evaluate.evaluate import _metrics_from_logits, _metrics_from_logits_multilabel
from trainer.BaseTrainer import BaseTrainer


class Unified_Trainer(BaseTrainer):
    """
    Unified Trainer implementing BaseTrainer framework.
    Supports both multiclass and multilabel classification tasks.

    Features:
    - Multiclass: CrossEntropyLoss with accuracy, F1, recall, precision metrics
    - Multilabel: BCEWithLogitsLoss with AUC and AP metrics
    - Memory and time tracking inherited from BaseTrainer
    """

    def __init__(self, args, logger, dataset, model, optimizer):
        """
        Initialize Unified Trainer.

        Args:
            args: Configuration arguments
            logger: Logger instance for logging
            dataset: Dataset object containing data loaders
            model: Neural network model to train
            optimizer: Optimizer for training
        """
        super().__init__(args, logger, dataset, model, optimizer)

    def _get_loss_function(self, task_type: str):
        """
        Get the appropriate loss function for the task.

        Args:
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Loss function
        """
        if task_type == 'multiclass':
            return nn.CrossEntropyLoss()
        elif task_type == 'multilabel':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

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
        if task_type == 'multiclass':
            acc, f1, rec, pre = _metrics_from_logits(y_true, y_logits)
            return {'Accuracy': acc, 'F1': f1, 'Recall': rec, 'Precision': pre}
        elif task_type == 'multilabel':
            auc, ap_macro = _metrics_from_logits_multilabel(y_true, y_logits)

            # Compute micro metrics as well
            auc_micro = roc_auc_score(y_true.reshape(-1), y_logits.reshape(-1))
            prec, rec, _ = precision_recall_curve(y_true.reshape(-1), y_logits.reshape(-1))
            ap_micro = auc(rec, prec)

            return {'AUC': auc}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

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
        self.model.train()
        ep_loss_sum, ep_batches = 0.0, 0
        ys, ylog = [], []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)

            # Use autocast with mixed precision
            autocast_enabled = (self.device.type == 'cuda')
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = self.model(self.dataset.data_o, inp)
                loss = loss_fct(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            ep_loss_sum += float(loss.item())
            ep_batches += 1

            # Store predictions for metrics computation
            ys.append(labels.detach().cpu())
            ylog.append(logits.detach().cpu())

        # Compute metrics
        if task_type == 'multiclass':
            y_true = torch.cat(ys, 0).numpy()
            y_log = torch.cat(ylog, 0).numpy()
        else:  # multilabel
            y_true = np.concatenate([y.cpu().numpy() for y in ys], axis=0)
            y_log = np.concatenate([y.cpu().numpy() for y in ylog], axis=0)

        metrics = self._compute_metrics(y_true, y_log, task_type)
        avg_loss = ep_loss_sum / max(ep_batches, 1)

        return metrics, avg_loss

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
        self.model.eval()
        ys, ylog, loss_sum, n = [], [], 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp, task_type)

                # Use autocast with mixed precision
                autocast_enabled = (self.device.type == 'cuda')
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    logits = self.model(self.dataset.data_o, inp)
                    loss = loss_fct(logits, labels)

                loss_sum += float(loss.item())
                n += 1

                # Store predictions for metrics computation
                ys.append(labels.detach().cpu())
                ylog.append(logits.detach().cpu())

        # Compute metrics
        if task_type == 'multiclass':
            y_true = torch.cat(ys, 0).numpy()
            y_log = torch.cat(ylog, 0).numpy()
        else:  # multilabel
            y_true = np.concatenate([y.cpu().numpy() for y in ys], axis=0)
            y_log = np.concatenate([y.cpu().numpy() for y in ylog], axis=0)

        metrics = self._compute_metrics(y_true, y_log, task_type)
        avg_loss = loss_sum / max(n, 1)

        return metrics, avg_loss

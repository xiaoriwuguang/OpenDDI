# MRCGNN_Trainer.py
import copy
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, precision_recall_curve, auc
)
from typing import Dict, Any, Tuple, Optional
from evaluate.evaluate import _metrics_from_logits, plot_metrics,_metrics_from_logits_multilabel

# Enable TF32 acceleration for A100 GPUs
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
from trainer.BaseTrainer import BaseTrainer


class MRCGNN_Trainer(BaseTrainer):
    """
    MRCGNN Trainer implementing BaseTrainer framework with MRCGNN-specific logic.
    
    Features:
    - Triple loss training (main loss + two auxiliary losses)
    - BCEWithLogitsLoss for auxiliary losses
    - Support for multiple data objects (data_o, data_s, data_a)
    """
    
    def __init__(self, args, logger, dataset, model, optimizer):
        """
        Initialize MRCGNN Trainer.

        Args:
            args: Configuration arguments
            logger: Logger instance for logging
            dataset: Dataset object containing data loaders
            model: Neural network model to train
            optimizer: Optimizer for training
        """
        super().__init__(args, logger, dataset, model, optimizer)
        self.time = time.time()
        # MRCGNN-specific loss function
        self.b_xent = nn.BCEWithLogitsLoss()

    def _get_loss_function(self, task_type: str):
        """
        Get the loss function for MRCGNN training.

        Args:
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Loss function
        """
        if task_type == 'multiclass':
            return nn.CrossEntropyLoss()
        else:
            return nn.BCEWithLogitsLoss()

    def _compute_metrics(self, y_true: np.ndarray, y_logits: np.ndarray,
                        task_type: str) -> Dict[str, float]:
        """
        Compute evaluation metrics for MRCGNN predictions.

        Args:
            y_true: Ground truth labels
            y_logits: Model logits
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Dictionary of computed metrics
        """
        if task_type == 'multilabel':
            auc, ap = _metrics_from_logits_multilabel(y_true, y_logits)
            return {'AUC': auc}
        else:
            acc, f1, rec, pre = _metrics_from_logits(y_true, y_logits)
            return {'Accuracy': acc, 'F1': f1, 'Recall': rec, 'Precision': pre}

    def _train_epoch(self, epoch: int, loss_fct, scaler: torch.cuda.amp.GradScaler,
                    task_type: str) -> Tuple[Dict[str, float], float]:
        """
        Train for one epoch with MRCGNN-specific logic.

        Args:
            epoch: Current epoch number
            loss_fct: Loss function
            scaler: GradScaler for mixed precision
            task_type: Type of task

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        y_pred_logits_epoch = []
        y_true_epoch = []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                # MRCGNN-specific model call - passing 4 parameters
                output, cla_os, cla_os_a, _ = self.model(
                    data_o=self.dataset.data_o,
                    data_s=self.dataset.data_s,
                    data_a=self.dataset.data_a,
                    idx=inp
                )

                # MRCGNN-specific triple loss
                loss1 = loss_fct(output, labels.long() if task_type == 'multiclass' else labels)
                # Ensure all tensors are on the same device
                data_a_y = self.dataset.data_a.y.float().to(self.device)
                loss2 = self.b_xent(cla_os, data_a_y)
                loss3 = self.b_xent(cla_os_a, data_a_y)
                loss_train = (self.args.loss_ratio1 * loss1 +
                            self.args.loss_ratio2 * loss2 +
                            self.args.loss_ratio3 * loss3)

            scaler.scale(loss_train).backward()
            scaler.step(self.optimizer)
            scaler.update()

            epoch_loss_sum += float(loss_train.item())
            epoch_batches += 1

            if task_type == 'multiclass':
                y_true_epoch.extend(labels.detach().cpu().numpy().tolist())
            else:
                y_true_epoch.append(labels.detach().cpu().numpy())
            y_pred_logits_epoch.append(output.detach().cpu().numpy())

        # Compute metrics
        if task_type == 'multiclass':
            y_true_epoch = np.array(y_true_epoch)
            y_pred_logits_epoch = np.concatenate(y_pred_logits_epoch, axis=0)
        else:
            y_true_epoch = np.concatenate(y_true_epoch, axis=0)
            y_pred_logits_epoch = np.concatenate(y_pred_logits_epoch, axis=0)

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        metrics = self._compute_metrics(y_true_epoch, y_pred_logits_epoch, task_type)

        return metrics, avg_loss

    def _evaluate(self, loader, loss_fct, task_type: str) -> Tuple[Dict[str, float], float]:
        """
        Evaluate model on given data loader.

        Args:
            loader: Data loader for evaluation
            loss_fct: Loss function
            task_type: Type of task

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        self.model.eval()
        y_pred_logits = []
        y_label = []
        loss_sum = 0.0
        batches = 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp, task_type)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    output, cla_os, cla_os_a, _ = self.model(
                        data_o=self.dataset.data_o,
                        data_s=self.dataset.data_s,
                        data_a=self.dataset.data_a,
                        idx=inp
                    )

                    loss1 = loss_fct(output, labels.long() if task_type == 'multiclass' else labels)
                    # Ensure all tensors are on the same device
                    data_a_y = self.dataset.data_a.y.float().to(self.device)
                    loss2 = self.b_xent(cla_os, data_a_y)
                    loss3 = self.b_xent(cla_os_a, data_a_y)
                    loss = (self.args.loss_ratio1 * loss1 +
                            self.args.loss_ratio2 * loss2 +
                            self.args.loss_ratio3 * loss3)

                loss_sum += float(loss.item())
                batches += 1

                if task_type == 'multiclass':
                    y_label.extend(labels.detach().cpu().numpy().tolist())
                else:
                    y_label.append(labels.detach().cpu().numpy())
                y_pred_logits.append(output.detach().cpu().numpy())

        # Compute metrics
        if task_type == 'multiclass':
            y_label_np = np.array(y_label)
            y_pred_logits_np = np.concatenate(y_pred_logits, axis=0)
        else:
            y_label_np = np.concatenate(y_label, axis=0)
            y_pred_logits_np = np.concatenate(y_pred_logits, axis=0)

        avg_loss = loss_sum / max(batches, 1)
        metrics = self._compute_metrics(y_label_np, y_pred_logits_np, task_type)

        return metrics, avg_loss

    def _move_data_to_device(self):
        """Override to handle MRCGNN-specific data objects."""
        if hasattr(self.dataset, 'data_o'):
            self.dataset.data_o = self.dataset.data_o.to(self.device)
        if hasattr(self.dataset, 'data_s'):
            self.dataset.data_s = self.dataset.data_s.to(self.device)
        if hasattr(self.dataset, 'data_a'):
            self.dataset.data_a = self.dataset.data_a.to(self.device)

        # Ensure model is also on the correct device
        if not next(self.model.parameters()).is_cuda and self.device.type == 'cuda':
            self.model.to(self.device)
        elif next(self.model.parameters()).is_cuda and self.device.type == 'cpu':
            self.model.to(self.device)
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

# —— A100 加速：开启 TF32（不影响数值打印，仅提速）——
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
from trainer.BaseTrainer import BaseTrainer


class TIGER_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)
        self.time = time.time()

    def _prepare_batch_data(self, batch_data, task_type: str = 'multiclass') -> torch.Tensor:

        labels = batch_data[4]

        if task_type == 'multiclass':
            return torch.as_tensor(np.array(labels), dtype=torch.long, device=self.device)
        elif task_type == 'multilabel':
            return torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _train_multiclass(self):
        print('Start Training (multiclass)...')

        scaler = self._setup_scaler()
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, scaler, 'multiclass')

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, 'multiclass')

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, 'multiclass')

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[Model] Test {metrics_str}")
        self._save_results(test_metrics, "Model")

    def _train_multilabel(self):
        print('Start Training (multilabel)...')

        scaler = self._setup_scaler()
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, scaler, 'multilabel')

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, 'multilabel')

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, 'multilabel')

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[Model] Test {metrics_str}")
        self._save_results(test_metrics, "Model")

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
        if task_type == 'multilabel':
            auc, ap = _metrics_from_logits_multilabel(y_true, y_logits)
            return {'AUC': auc}
        else:
            acc, f1, rec, pre = _metrics_from_logits(y_true, y_logits)
            return {'Accuracy': acc, 'F1': f1, 'Recall': rec, 'Precision': pre}

    def _train_epoch(self, epoch: int, scaler: torch.cuda.amp.GradScaler,
                    task_type: str) -> Tuple[Dict[str, float], float]:

        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        y_pred_logits_epoch = []
        y_true_epoch = []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                output = self.model(inp)
                loss_train = self.model.loss(output, labels.long() if task_type == 'multiclass' else labels) 

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

    def _evaluate(self, loader, task_type: str) -> Tuple[Dict[str, float], float]:
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
                    output= self.model(inp)
                    loss = self.model.loss(output, labels.long() if task_type == 'multiclass' else labels)
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

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


class MRCGNN_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)
        self.time = time.time()
        # MRCGNN特有的损失函数
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
                # MRCGNN特有的模型调用 - 传递4个参数
                output, cla_os, cla_os_a, _ = self.model(
                    data_o=self.dataset.data_o,
                    data_s=self.dataset.data_s,
                    data_a=self.dataset.data_a,
                    idx=inp
                )

                # MRCGNN特有的三重损失
                loss1 = loss_fct(output, labels.long() if task_type == 'multiclass' else labels)
                # 确保所有张量都在相同设备上
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
                    # 确保所有张量都在相同设备上
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

        # 确保模型也在正确的设备上
        if not next(self.model.parameters()).is_cuda and self.device.type == 'cuda':
            self.model.to(self.device)
        elif next(self.model.parameters()).is_cuda and self.device.type == 'cpu':
            self.model.to(self.device)

    # def train(self):
    #     """Override base train method to call specific methods."""
    #     if self.args.matrix in ['multilabel', 'twosides']:
    #         self._train_multilabel()
    #     else:
    #         self._train_multi()

    # def _train_multi(self):
    #     """Legacy method - redirects to base multiclass training."""
    #     self._train_multiclass()

    # def _train_multilabel(self):
    #     """Legacy method - redirects to base multilabel training."""
    #     pass

    # # Legacy methods for backward compatibility
    # def test_multi(self, printfou):
    #     """Legacy method for backward compatibility."""
    #     task_type = 'multiclass'
    #     loss_fct = self._get_loss_function(task_type)
    #     loader = self.dataset.val_loader if printfou == 0 else self.dataset.test_loader
    #     metrics, avg_loss = self._evaluate(loader, loss_fct, task_type)

    #     # Legacy file output
    #     if printfou == 1:
    #         with open(getattr(self.args, 'out_file', 'result.txt'), 'a') as f:
    #             f.write('111  ' + str(metrics['Accuracy']) + '   ' + str(metrics['F1']) +
    #                     '   ' + str(metrics['Recall']) + '   ' + str(metrics['Precision']) +
    #                     '   ' + 'time:' + str(time.time() - self.time) + '   ' +
    #                     'memory:' + str(torch.cuda.max_memory_allocated() / (1024 ** 2)) + '\n')

    #     return metrics['Accuracy'], metrics['F1'], metrics['Recall'], metrics['Precision'], avg_loss

    # def test_multilabel(self, printfou):
    #     """Legacy method for backward compatibility."""
    #     task_type = 'multilabel'
    #     loss_fct = self._get_loss_function(task_type)
    #     loader = self.dataset.val_loader if printfou == 0 else self.dataset.test_loader
    #     metrics, avg_loss = self._evaluate(loader, loss_fct, task_type)

    #     # Legacy file output
    #     if printfou == 1:
    #         with open(getattr(self.args, 'out_file', 'result.txt'), 'a') as f:
    #             f.write('111  ' + str(metrics['AUC']) + '   ' + str(metrics['AP']) + '\n')

    #     return metrics['AUC'], avg_loss
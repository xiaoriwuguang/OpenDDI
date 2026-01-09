import copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate.evaluate import _metrics_from_logits, _metrics_from_logits_multilabel
from trainer.BaseTrainer import BaseTrainer
from typing import Dict, Any, Tuple, Optional

class ZeroDDI_Trainer(BaseTrainer):
    """
    ZeroDDI: 双模态统一对齐 (DUA)
    损失 = λ1*对齐 + λ2*均匀化(pair) + λ3*均匀化(event)
    """
    def __init__(self, args, logger, dataset, model, optimizer=None):
        super().__init__(args, logger, dataset, model, optimizer)

        # 绑定图和事件语义（一次性）
        self.model.bind_graph(self.dataset.data_graph)
        self.model.update_event_U(self.dataset.event_sem)

        # 判断任务类型
        self.multi_label = getattr(self.args, "matrix", None) in ['multilabel', 'twosides']

        # 系数
        self.lambda_align   = float(getattr(self.args, "lambda_align", 1.0))
        self.lambda_u_pair  = float(getattr(self.args, "lambda_u_pair", 0.1))
        self.lambda_u_event = float(getattr(self.args, "lambda_u_event", 0.1))
        self.uniform_t      = float(getattr(self.args, "uniform_t", 2.0))

    # ----------------- uniformity 正则 -----------------
    def _uniformity_loss(self, X, t=2.0, max_pairs=4096):
        B = X.size(0)
        if B < 2: return X.new_zeros(())
        num = min(max_pairs, B*(B-1)//2)
        idx_i = torch.randint(0, B, (num,), device=X.device)
        idx_j = torch.randint(0, B, (num,), device=X.device)
        diff = (X[idx_i]-X[idx_j]).pow(2).sum(dim=1)
        return torch.log(torch.exp(-t*diff).mean() + 1e-12)

    def _get_loss_function(self, task_type: str):
        """
        Get the loss function for ZeroDDI trainer.

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
        Compute evaluation metrics for ZeroDDI predictions.

        Args:
            y_true: Ground truth labels
            y_logits: Model logits
            task_type: Type of task ('multiclass' or 'multilabel')

        Returns:
            Dictionary of computed metrics
        """
        if task_type == 'multilabel':
            # 使用多标签评估函数
            auc, ap = _metrics_from_logits_multilabel(y_true, y_logits)
            return {'AUC': auc}
        else:
            # 使用多分类评估函数
            acc, f1, rec, pre = _metrics_from_logits(y_true, y_logits)
            return {
                'Accuracy': acc,
                'F1': f1,
                'Recall': rec,
                'Precision': pre
            }

    def _train_epoch(self, epoch: int, loss_fct, scaler: torch.cuda.amp.GradScaler,
                    task_type: str) -> Tuple[Dict[str, float], float]:
        """
        Train for one epoch with ZeroDDI-specific logic.

        Args:
            epoch: Current epoch number
            loss_fct: Loss function
            scaler: GradScaler for mixed precision
            task_type: Type of task

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        self.model.train()
        ep_loss, y_true_ep, y_log_ep, batches = 0.0, [], [], 0

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                logits, z = self.model(None, inp)

                # 根据任务类型正确处理标签格式
                if task_type == 'multiclass':
                    # 多分类任务需要long类型的标签
                    loss_align = loss_fct(logits, labels.long())
                else:
                    # 多标签任务保持原始标签格式
                    loss_align = loss_fct(logits, labels)
                loss_u_pair = self._uniformity_loss(z, t=self.uniform_t)
                U = self.model.U.detach()
                loss_u_event = self._uniformity_loss(U, t=self.uniform_t)

                loss = (self.lambda_align*loss_align +
                        self.lambda_u_pair*loss_u_pair +
                        self.lambda_u_event*loss_u_event)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            ep_loss += float(loss.item())
            batches += 1
            y_true_ep.append(labels.detach().cpu())
            y_log_ep.append(logits.detach().cpu())

        # Compute metrics
        y_true_np = torch.cat(y_true_ep, 0).numpy()
        y_log_np = torch.cat(y_log_ep, 0).numpy()
        avg_loss = ep_loss / max(batches, 1)
        metrics = self._compute_metrics(y_true_np, y_log_np, task_type)

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
        ys, logits_all = [], []
        loss_sum, batches = 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp, task_type)

                logits, z = self.model(None, inp)
                # 根据任务类型正确处理标签格式
                if task_type == 'multiclass':
                    # 多分类任务需要long类型的标签
                    loss_align = loss_fct(logits, labels.long())
                else:
                    # 多标签任务保持原始标签格式
                    loss_align = loss_fct(logits, labels)
                loss = self.lambda_align * loss_align

                loss_sum += float(loss.item())
                batches += 1
                ys.append(labels.detach().cpu())
                logits_all.append(logits.detach().cpu())

        y_true = torch.cat(ys, dim=0).numpy()
        y_log = torch.cat(logits_all, dim=0).numpy()
        avg_loss = loss_sum / max(batches, 1)
        metrics = self._compute_metrics(y_true, y_log, task_type)

        return metrics, avg_loss

    def _evaluate_loader(self, loader):
        """Legacy method for backward compatibility."""
        task_type = 'multilabel' if self.multi_label else 'multiclass'
        metrics, avg_loss = self._evaluate(loader, self.loss_align, task_type)
        return metrics['Accuracy'], metrics['F1'], metrics['Recall'], metrics['Precision'], avg_loss

    def train(self):
        """Override base train method with ZeroDDI-specific logic."""
        print("Start Training (ZeroDDI)...")

        # Move data to device using base class method
        self._move_data_to_device()

        # Determine task type
        if self.args.matrix in ['multilabel', 'twosides']:
            self._train_multilabel()
        else:
            self._train_multiclass()

    def _train_multiclass(self):
        """Training loop for multiclass classification with ZeroDDI logic."""
        task_type = 'multiclass'
        loss_fct = self._get_loss_function(task_type)
        scaler = self._setup_scaler()

        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, loss_fct, scaler, task_type)

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, loss_fct, task_type)

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, loss_fct, task_type)

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[ZeroDDI] Test {metrics_str}")
        self._save_results(test_metrics, "ZeroDDI")

        # Legacy file output - 根据任务类型选择不同的输出格式
        if task_type == 'multilabel':
            with open(getattr(self.args, "out_file", "result.txt"), "a") as f:
                f.write(f"ZeroDDI {test_metrics.get('AUC', 0.0)} {test_metrics.get('AP', 0.0)}\n")
        else:
            with open(getattr(self.args, "out_file", "result.txt"), "a") as f:
                f.write(f"ZeroDDI {test_metrics['Accuracy']} {test_metrics['F1']} {test_metrics['Recall']} {test_metrics['Precision']}\n")

    def _train_multilabel(self):
        """Training loop for multilabel classification with ZeroDDI logic."""
        task_type = 'multilabel'
        loss_fct = self._get_loss_function(task_type)
        scaler = self._setup_scaler()

        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics, train_loss = self._train_epoch(epoch, loss_fct, scaler, task_type)

            # Validate
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader, loss_fct, task_type)

            # Log progress
            self._log_training_progress(epoch,
                                       {'Loss': train_loss, **train_metrics},
                                       {'Loss': val_loss, **val_metrics})

        # Final test evaluation
        self.model.eval()
        test_metrics, test_loss = self._evaluate(self.dataset.test_loader, loss_fct, task_type)

        # Print and save results
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        print(f"[ZeroDDI] Test {metrics_str}")
        self._save_results(test_metrics, "ZeroDDI")

        # Legacy file output - 根据任务类型选择不同的输出格式
        if task_type == 'multilabel':
            with open(getattr(self.args, "out_file", "result.txt"), "a") as f:
                f.write(f"ZeroDDI {test_metrics.get('AUC', 0.0)} {test_metrics.get('AP', 0.0)}\n")
        else:
            with open(getattr(self.args, "out_file", "result.txt"), "a") as f:
                f.write(f"ZeroDDI {test_metrics['Accuracy']} {test_metrics['F1']} {test_metrics['Recall']} {test_metrics['Precision']}\n")

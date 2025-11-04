import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, precision_recall_curve, auc
)
from matplotlib import pyplot as plt

def _metrics_from_logits(y_true_np, y_logits_np):
    """返回 Acc/F1/Recall/Precision（macro，且避免未定义警告）"""
    y_pred_np = np.argmax(y_logits_np, axis=1)
    acc = accuracy_score(y_true_np, y_pred_np)
    f1  = f1_score(y_true_np, y_pred_np, average='macro',   zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    pre = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    return acc, f1, rec, pre

def _metrics_from_logits_multilabel(y_true_np, y_logits_np):
    """返回 AUC 和 AP（macro 平均，避免未定义警告）"""
    auc_macro = 0.0
    ap_macro = 0.0
    valid_k = y_true_np.shape[1]  # 标签数量
    for k in range(y_true_np.shape[1]):
        if np.sum(y_true_np[:, k]) < 1 or np.sum(y_true_np[:, k]) == len(y_true_np[:, k]):
            valid_k -= 1  # 跳过全0或全1的标签
            continue
        auc_macro += roc_auc_score(y_true_np[:, k], y_logits_np[:, k])
        ap_macro += average_precision_score(y_true_np[:, k], y_logits_np[:, k])
    auc_macro = auc_macro / valid_k if valid_k > 0 else 0.0
    ap_macro = ap_macro / valid_k if valid_k > 0 else 0.0
    return roc_auc_score(y_true_np.reshape(-1), y_logits_np.reshape(-1)), ap_macro

def plot_metrics(train_metrics, val_metrics, metric_name, out_file_prefix):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}')
    plt.xlabel('Epoch'); plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs. Epoch'); plt.legend(); plt.grid(True)
    plt.savefig(f'{out_file_prefix}_{metric_name.lower().replace(" ", "_")}.png')
    plt.close()
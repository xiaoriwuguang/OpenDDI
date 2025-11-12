import os
import numpy as np
import pandas as pd
import torch
import argparse
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from data.BaseDataset import BaseDataset

def _as_path_list(maybe_list) -> List[str]:
    """Convert input to list of paths."""
    if isinstance(maybe_list, (list, tuple)): return list(maybe_list)
    if isinstance(maybe_list, str): return [p.strip() for p in maybe_list.split(',') if p.strip()]
    return []

def _read_csv_embedding(path: str) -> Tuple[dict, int]:
    """Read CSV embedding file and return dictionary and dimension."""
    df = pd.read_csv(path)
    id_col = df.columns[0]
    ids = df[id_col].astype(str).tolist()
    vecs = df.drop(columns=[id_col]).to_numpy(dtype=np.float32)
    dim = vecs.shape[1]
    return {ids[i]: vecs[i] for i in range(len(ids))}, dim

def _merge_id2vec(dicts_dims: List[Tuple[dict, int]]) -> Tuple[dict, int]:
    """Merge multiple embedding dictionaries into one."""
    all_ids = set()
    for d, _ in dicts_dims: all_ids |= set(d.keys())
    all_ids = sorted(list(all_ids))
    total_dim = sum(dim for _, dim in dicts_dims)
    merged = {}
    for id_ in all_ids:
        parts = []
        for d, dim in dicts_dims:
            parts.append(d[id_] if id_ in d else np.zeros(dim, dtype=np.float32))
        merged[id_] = np.concatenate(parts, axis=0).astype(np.float32)
    return merged, total_dim

def read_embeddings_any(paths: List[str]) -> Tuple[dict, int]:
    """
    Read embeddings from multiple files of different formats.
    
    Args:
        paths: List of file paths to read from
        
    Returns:
        Tuple of (embedding dictionary, total dimension)
    """
    dicts_dims = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == '.pt':
            data = torch.load(p)
            if not isinstance(data, dict):
                raise ValueError(f"{p} 不是 dict 格式的 .pt")
            cur = {str(k): (v.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(v) else np.asarray(v, dtype=np.float32))
                   for k, v in data.items()}
            any_key = next(iter(cur.keys()))
            dim = cur[any_key].shape[0]
            dicts_dims.append((cur, dim))
        elif ext == '.csv':
            cur, dim = _read_csv_embedding(p)
            dicts_dims.append((cur, dim))
        else:
            raise ValueError(f"不支持的嵌入文件后缀：{p}")
    return _merge_id2vec(dicts_dims)

class ZeroDDI_dataset(BaseDataset):
    """
    ZeroDDI dataset class with support for multi-modal node features and zero-shot learning.
    
    Features:
    - Node modalities (supports concatenation of multiple CSV/PT files)
    - Training set graph construction (DDI graph)
    - Supports regular/multi-label classification
    - Feature and label noise injection during loading phase (training set only)
    - Zero-shot learning protocols (CZSL, GZSL)
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize ZeroDDI dataset.
        
        Args:
            args: Namespace object containing configuration parameters
        """
        super().__init__(args)
        # ZeroDDI specific attributes
        self.seen_classes = self.unseen_classes = None
        self.event_sem = None  # (K, d_e)

    # ---------- Event semantics ----------
    def _load_event_semantics(self, K: int):
        """Load event semantics for zero-shot learning."""
        path = getattr(self.args, 'event_sem_path', None)
        E = None
        if path and os.path.isfile(path):
            if path.lower().endswith('.npy'):
                E = np.load(path).astype(np.float32)
            elif path.lower().endswith('.csv'):
                num_df = pd.read_csv(path).select_dtypes(include=[np.number])
                E = num_df.to_numpy(dtype=np.float32)
        if E is None or E.shape[0] != K:
            E = np.eye(K, dtype=np.float32)
        self.event_sem = torch.tensor(E, dtype=torch.float32)
        self.args.event_sem_dim = E.shape[1]

    # ---------- ✨ Main entry: loading + noise + split + graph construction ----------
    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """
        Main data loading method.
        
        Args:
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        # First call parent class's load_data for basic data loading logic
        super().load_data(val_ratio, test_ratio)

        # Then perform ZeroDDI-specific processing
        self._load_zero_ddi_specific()

    def _load_zero_ddi_specific(self):
        """
        ZeroDDI-specific data loading logic including zero-shot protocol and event semantics.
        """
        # Check if zero-shot protocol is used
        protocol = str(getattr(self.args, 'zs_protocol', 'none')).upper()
        if protocol not in ('NONE', 'CZSL', 'GZSL'):
            protocol = 'NONE'

        # Only apply zero-shot processing to multiclass tasks
        if protocol != 'NONE' and self.args.matrix not in ['multilabel', 'twosides']:
            self._handle_zero_shot_protocol()

        # Load event semantics
        K = getattr(self.args, 'num_classes', 0)
        self._load_event_semantics(K)

        # Ensure data_graph points to the correct data object
        self.data_graph = self.data_o

        print(f"[ZeroDDI_dataset] X_dim={self.args.dimensions}, K={K}, "
              f"protocol={protocol}, "
              f"seen={None if self.seen_classes is None else len(self.seen_classes)}, "
              f"unseen={None if self.unseen_classes is None else len(self.unseen_classes)}")

    def _handle_zero_shot_protocol(self):
        """Handle zero-shot learning protocol."""
        rng = np.random.RandomState(getattr(self.args, 'zs_seed', 1))

        # Get current training data
        if self.train_loader and hasattr(self.train_loader, 'dataset'):
            train_data = np.array(self.train_loader.dataset.triple)
            y = train_data[:, 2]
            classes = np.unique(y)

            zs_ratio = float(getattr(self.args, 'zs_ratio', 0.3))
            num_unseen = max(1, int(round(len(classes) * zs_ratio)))
            unseen = rng.choice(classes, size=num_unseen, replace=False)
            seen = np.array([c for c in classes if c not in set(unseen)])

            # Update seen_classes and unseen_classes
            self.seen_classes = seen
            self.unseen_classes = unseen
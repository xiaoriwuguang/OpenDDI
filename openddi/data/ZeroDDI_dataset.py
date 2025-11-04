import os
import numpy as np
import pandas as pd
import torch
import argparse
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from data.datasetTool import (
    Base_multi_dataset, Base_multilabel_dataset,
    _read_multi_pairs_and_remap, _read_multilabel_pairs_and_remap
)
from data.BaseDataset import BaseDataset

def _as_path_list(maybe_list) -> List[str]:
    if isinstance(maybe_list, (list, tuple)): return list(maybe_list)
    if isinstance(maybe_list, str): return [p.strip() for p in maybe_list.split(',') if p.strip()]
    return []

def _read_csv_embedding(path: str) -> Tuple[dict, int]:
    df = pd.read_csv(path)
    id_col = df.columns[0]
    ids = df[id_col].astype(str).tolist()
    vecs = df.drop(columns=[id_col]).to_numpy(dtype=np.float32)
    dim = vecs.shape[1]
    return {ids[i]: vecs[i] for i in range(len(ids))}, dim

def _merge_id2vec(dicts_dims: List[Tuple[dict, int]]) -> Tuple[dict, int]:
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
    节点模态（可拼接多模态 CSV/PT） + 训练集构图（DDI 图）
    支持：常规/多标签；并在**加载阶段**注入特征噪声与标签噪声（仅训练集）
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # ZeroDDI特有的属性
        self.seen_classes = self.unseen_classes = None
        self.event_sem = None  # (K, d_e)

    # ---------- 1) 节点多模态 ----------
    def _load_node_embeddings(self):
        emb_paths = _as_path_list(getattr(self.args, 'embedding_path', []))
        if not emb_paths:
            raise ValueError("请通过 --embedding_path 提供一个或多个嵌入文件（.pt 或 .csv，可逗号分隔）")
        id2vec, emb_dim = read_embeddings_any(emb_paths)
        return id2vec, emb_dim

    # ---------- 2) 读取配对与标签 ----------
    def _read_pairs_labels(self):
        if self.args.matrix in ["multilabel", "twosides"]:
            pairs_df, num_ddi = _read_multilabel_pairs_and_remap(self.args.matrix_path)
            return 'multilabel', pairs_df, int(num_ddi)
        else:
            pairs_df, num_rel = _read_multi_pairs_and_remap(self.args.matrix_path)
            return 'multiclass', pairs_df, int(num_rel)

    # ---------- 3) 常规划分（或零样本协议） ----------
    def _split_normal_or_zs(self, triples: np.ndarray, mode: str):
        protocol = str(getattr(self.args, 'zs_protocol', 'none')).upper()
        rng = np.random.RandomState(getattr(self.args, 'zs_seed', 1))
        if protocol not in ('NONE', 'CZSL', 'GZSL'): protocol = 'NONE'

        if protocol == 'NONE' or mode == 'multilabel':
            idx = rng.permutation(len(triples))
            n = len(triples); n_test = int(n * getattr(self.args, 'test_ratio', 0.2)); n_val = int(n * getattr(self.args, 'val_ratio', 0.1))
            return triples[idx[n_test+n_val:]], triples[idx[n_test:n_test+n_val]], triples[idx[:n_test]], None, None

        # —— 零样本，仅 multiclass
        y = triples[:, 2]
        classes = np.unique(y)
        zs_ratio = float(getattr(self.args, 'zs_ratio', 0.3))
        num_unseen = max(1, int(round(len(classes) * zs_ratio)))
        unseen = rng.choice(classes, size=num_unseen, replace=False)
        seen = np.array([c for c in classes if c not in set(unseen)])

        seen_mask = np.isin(y, seen)
        unseen_mask = np.isin(y, unseen)

        train_all = triples[seen_mask]
        cand = triples if protocol == 'GZSL' else triples[unseen_mask]

        idx = rng.permutation(len(cand))
        n = len(cand); n_test = int(n * getattr(self.args, 'test_ratio', 0.2)); n_val = int(n * getattr(self.args, 'val_ratio', 0.1))
        test = cand[idx[:n_test]]; val = cand[idx[n_test:n_test+n_val]]; train = train_all

        self.seen_classes, self.unseen_classes = seen, unseen
        return train, val, test, seen, unseen

    # ---------- 4) 训练图 ----------
    def _build_graph_from_train(self, features_o: np.ndarray, train_triples: np.ndarray):
        use_ratio = float(getattr(self.args, "network_ratio", 1.0))
        if use_ratio <= 0 or use_ratio > 1: use_ratio = 1.0
        edges = train_triples
        if use_ratio < 1.0:
            keep = int(max(1, round(edges.shape[0] * use_ratio)))
            sel = np.random.RandomState(1).permutation(edges.shape[0])[:keep]
            edges = edges[sel]
            print(f"[graph] edge_ratio={use_ratio} -> {keep}/{train_triples.shape[0]} edges.")

        edge_index, edge_type = [], []
        for i, j, r in edges:
            edge_index.append([int(i), int(j)]); edge_type.append(int(r))
            edge_index.append([int(j), int(i)]); edge_type.append(int(r))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type  = torch.tensor(edge_type,  dtype=torch.long)
        x = torch.tensor(features_o, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)

    # ---------- 5) 事件语义 ----------
    def _load_event_semantics(self, K: int):
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

    # ---------- ✨ 入口：加载 + 噪声 + 划分 + 构图 ----------
    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        # 首先调用基类的load_data处理基本的数据加载逻辑
        super().load_data(val_ratio, test_ratio)

        # 然后进行ZeroDDI特有的处理
        self._load_zero_ddi_specific()

    def _load_zero_ddi_specific(self):
        """
        ZeroDDI特有的数据加载逻辑，包括零样本协议处理和事件语义加载
        """
        # 判断是否为零样本协议
        protocol = str(getattr(self.args, 'zs_protocol', 'none')).upper()
        if protocol not in ('NONE', 'CZSL', 'GZSL'):
            protocol = 'NONE'

        # 仅对多分类任务进行零样本处理
        if protocol != 'NONE' and self.args.matrix not in ['multilabel', 'twosides']:
            self._handle_zero_shot_protocol()

        # 加载事件语义
        K = getattr(self.args, 'num_classes', 0)
        self._load_event_semantics(K)

        # 确保data_graph指向正确的数据对象
        self.data_graph = self.data_o

        print(f"[ZeroDDI_dataset] X_dim={self.args.dimensions}, K={K}, "
              f"protocol={protocol}, "
              f"seen={None if self.seen_classes is None else len(self.seen_classes)}, "
              f"unseen={None if self.unseen_classes is None else len(self.unseen_classes)}")

    def _handle_zero_shot_protocol(self):
        """
        处理零样本学习协议
        """
        rng = np.random.RandomState(getattr(self.args, 'zs_seed', 1))

        # 获取当前训练数据
        if self.train_loader and hasattr(self.train_loader, 'dataset'):
            train_data = np.array(self.train_loader.dataset.triple)
            y = train_data[:, 2]
            classes = np.unique(y)

            zs_ratio = float(getattr(self.args, 'zs_ratio', 0.3))
            num_unseen = max(1, int(round(len(classes) * zs_ratio)))
            unseen = rng.choice(classes, size=num_unseen, replace=False)
            seen = np.array([c for c in classes if c not in set(unseen)])

            # 更新seen_classes和unseen_classes
            self.seen_classes = seen
            self.unseen_classes = unseen


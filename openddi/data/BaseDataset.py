import os
import re
import pandas as pd
import numpy as np
import torch
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


# =============================================================================
# Data Loading and Preprocessing Module
# =============================================================================

class DataLoadingModule:
    """
    Data loading and preprocessing module.
    
    Responsible for reading embedding files, paired data files, and basic data preprocessing.
    """

    @staticmethod
    def read_id_embedding_pt(embedding_paths: Union[str, List[str]]) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Read single or multiple modality embedding files and concatenate embedding vectors.

        Args:
            embedding_paths: Path or list of paths to embedding files.

        Returns:
            Tuple containing:
            - id2vec: Dictionary mapping IDs to concatenated embedding vectors
            - total_dim: Total dimension of concatenated embedding vectors
        """
        if isinstance(embedding_paths, str):
            embedding_paths = [embedding_paths]

        id2vec = {}
        total_dim = 0

        # Iterate through all modality embedding paths
        for path in embedding_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"未找到嵌入文件: {path}")

            # Read embedding file
            data = torch.load(path)
            if not isinstance(data, dict):
                raise ValueError(f"期望 pt 文件中是 dict，但得到的是 {type(data)}")

            # Convert to numpy arrays, ensuring float32
            current_id2vec = {str(k): v.detach().cpu().numpy().astype(np.float32) for k, v in data.items()}

            # Get current modality dimension
            example_key = next(iter(current_id2vec))
            current_dim = current_id2vec[example_key].shape[0]
            total_dim += current_dim

            # Initialize id2vec or concatenate embeddings
            if not id2vec:
                id2vec = current_id2vec
            else:
                # Ensure ID consistency
                if set(id2vec.keys()) != set(current_id2vec.keys()):
                    raise ValueError(f"嵌入文件 {path} 的 ID 集与之前的嵌入不一致")
                # Connect the embedding vectors of each ID
                for id_ in id2vec:
                    id2vec[id_] = np.concatenate([id2vec[id_], current_id2vec[id_]], axis=0)

        return id2vec, total_dim

    @staticmethod
    def read_id_embedding_pt_split(embedding_paths: List[str]) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
        """
        Read multiple modality embedding files and store embedding vectors separately.

        Args:
            embedding_paths: List of paths to embedding files.

        Returns:
            Tuple containing:
            - modal2id2vec: Dictionary mapping modality indices to id2vec dictionaries
            - dims: List of embedding vector dimensions for each modality
        """
        modal2id2vec = {}
        dims = []

        for idx, path in enumerate(embedding_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"未找到嵌入文件: {path}")

            data = torch.load(path)
            if not isinstance(data, dict):
                raise ValueError(f"期望 pt 文件中是 dict，但得到的是 {type(data)}")

            current_id2vec = {str(k): v.detach().cpu().numpy().astype(np.float32) for k, v in data.items()}
            example_key = next(iter(current_id2vec))
            current_dim = current_id2vec[example_key].shape[0]
            dims.append(current_dim)

            modal2id2vec[idx] = current_id2vec

        # Verify that ID sets are consistent across all modalities
        id_sets = [set(id2vec.keys()) for id2vec in modal2id2vec.values()]
        if len(set(frozenset(id_set) for id_set in id_sets)) > 1:
            raise ValueError("不同模态的嵌入文件中 ID 集不一致")

        return modal2id2vec, dims

    @staticmethod
    def read_multi_pairs_and_remap(matrix_path: str) -> Tuple[pd.DataFrame, int]:
        """
        Read multi-class paired data and perform label remapping.

        Args:
            matrix_path: Path to multi-class data file.

        Returns:
            Tuple containing:
            - df: Processed DataFrame containing id1, id2, ddi columns
            - num_relations: Number of relation types
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到多分类文件: {matrix_path}")

        df = pd.read_csv(matrix_path, dtype=str)
        cols_lower = {c.lower(): c for c in df.columns}
        for k in ['id1', 'id2', 'ddi']:
            if k not in cols_lower:
                raise KeyError(f"文件 {matrix_path} 缺少列：{k}")

        id1 = df['id1'].astype(str)
        id2 = df['id2'].astype(str)
        ddi = pd.to_numeric(df['ddi'], errors='coerce')

        df2 = pd.DataFrame({'id1': id1, 'id2': id2, 'ddi_raw': ddi})
        df2 = df2.dropna(subset=['id1', 'id2', 'ddi_raw']).copy()
        df2['ddi_raw'] = df2['ddi_raw'].astype(int)

        # Remap to continuous labels based on unique values
        unique_raw = np.sort(df2['ddi_raw'].unique())
        raw2new = {raw: i for i, raw in enumerate(unique_raw)}
        df2['ddi'] = df2['ddi_raw'].map(raw2new).astype(int)
        num_relations = len(unique_raw)

        return df2[['id1','id2','ddi']], num_relations

    @staticmethod
    def read_multilabel_pairs_and_remap(matrix_path: str) -> Tuple[pd.DataFrame, int]:
        """
        Read multi-label paired data.

        Args:
            matrix_path: Path to multi-label data file.

        Returns:
            Tuple containing:
            - df: Processed DataFrame containing id1, id2, ddi columns
            - num_ddi: Number of labels
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到多标签分类文件 {matrix_path}")

        df = pd.read_csv(matrix_path, dtype=str)
        cols_lower = {c.lower(): c for c in df.columns}
        for k in ['id1', 'id2', 'ddi']:
            if k not in cols_lower:
                raise KeyError(f"文件 {matrix_path} 缺少列：{k}")

        id1 = df['id1'].astype(str)
        id2 = df['id2'].astype(str)
        ddi = df['ddi'].apply(lambda x: np.array([int(i) for i in str(x).split(',')], dtype=np.float32))

        df2 = pd.DataFrame({'id1': id1, 'id2': id2, 'ddi': ddi})
        df2 = df2.dropna(subset=['id1', 'id2', 'ddi']).copy()

        num_ddi = len(df2['ddi'].iloc[0])
        return df2[['id1', 'id2', 'ddi']], num_ddi


# =============================================================================
# Feature Processing and Noise Injection Module
# =============================================================================

class FeatureProcessingModule:
    """
    Feature processing and noise injection module.
    
    Responsible for feature matrix construction, standardization, noise injection, etc.
    """

    @staticmethod
    def build_feature_matrix(drug_list: List[str], id2vec: Dict[str, np.ndarray],
                           emb_dim: int, args: argparse.Namespace) -> torch.Tensor:
        """
        Build drug feature matrix.

        Args:
            drug_list: List of drug IDs.
            id2vec: Dictionary mapping IDs to embedding vectors.
            emb_dim: Embedding dimension.
            args: Configuration parameters.

        Returns:
            Standardized feature matrix.
        """
        feats, miss = [], 0
        for d in drug_list:
            if d in id2vec:
                v = id2vec[d]
            elif d.upper() in id2vec:
                v = id2vec[d.upper()]
            elif d.lower() in id2vec:
                v = id2vec[d.lower()]
            else:
                miss += 1
                v = np.zeros(emb_dim, dtype=np.float32)
            feats.append(v)

        feats = np.asarray(feats, dtype=np.float32)

        # Feature standardization
        feats = FeatureProcessingModule.normalize_features(feats)

        # Add Gaussian noise
        if getattr(args, 'noise_std', 0.0) > 0:
            feats = FeatureProcessingModule.add_gaussian_noise(feats, args.noise_std)

        # Sparse dropout
        if float(getattr(args, 'sparse_drop_rate', 0.0)) > 0:
            feats = FeatureProcessingModule.sparse_dropout(feats, args.sparse_drop_rate)

        return torch.tensor(feats, dtype=torch.float32)

    @staticmethod
    def normalize_features(feats: np.ndarray) -> np.ndarray:
        """
        Standardize features.

        Args:
            feats: Original feature matrix.

        Returns:
            Standardized feature matrix.
        """
        return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    @staticmethod
    def add_gaussian_noise(feats: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Add Gaussian noise.

        Args:
            feats: Original feature matrix.
            noise_std: Noise standard deviation.

        Returns:
            Feature matrix with added noise.
        """
        noise = np.random.normal(0, float(noise_std), feats.shape).astype(np.float32)
        return feats + noise

    @staticmethod
    def sparse_dropout(feats: np.ndarray, drop_rate: float) -> np.ndarray:
        """
        Apply sparse dropout.

        Args:
            feats: Original feature matrix.
            drop_rate: Dropout rate.

        Returns:
            Sparsified feature matrix.
        """
        mask = (np.random.rand(*feats.shape) > drop_rate).astype(np.float32)
        return feats * mask


# =============================================================================
# Data Splitting and Sampling Module
# =============================================================================

class DataSplittingModule:
    """
    Data splitting and sampling module.
    
    Responsible for dataset splitting, label noise injection, sparse sampling, etc.
    """

    @staticmethod
    def split_data(triples: np.ndarray, val_ratio: float = 0.1,
                  test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset.

        Args:
            triples: Triple data.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            random_seed: Random seed.

        Returns:
            Tuple containing:
            - train_data: Training set
            - val_data: Validation set
            - test_data: Test set
        """
        rng = np.random.RandomState(random_seed)
        rng.shuffle(triples)

        n_total = len(triples)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_data = triples[:n_test]
        val_data = triples[n_test:n_test+n_val]
        train_data = triples[n_test+n_val:]

        print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")
        return train_data, val_data, test_data

    @staticmethod
    def _entity_holdout_split_indices(triples: np.ndarray, val_ratio: float = 0.1,
                                      test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset based on drug entities to ensure some drugs in test set do not appear in train/validation set.

        Args:
            triples (List[Tuple]): Triple data (h, t, r).
            val_ratio (float): Validation set ratio (applied to non-test part).
            test_ratio (float): Test set ratio (target proportion of total samples).
            random_seed (int): Random seed.

        Returns:
            Tuple[List[int], List[int], List[int]]: 
                - train_idx: Indices for training set
                - val_idx: Indices for validation set
                - test_idx: Indices for test set
        """
        n_total = len(triples)
        if n_total == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        rng = np.random.RandomState(random_seed)
        target_test = max(1, int(np.ceil(n_total * float(test_ratio))))

        # Count the sample indices associated with each drug.
        drug_to_indices: Dict[int, set] = defaultdict(set)
        for idx, (h, t, _) in enumerate(triples):
            drug_to_indices[int(h)].add(idx)
            drug_to_indices[int(t)].add(idx)

        drug_ids = np.array(list(drug_to_indices.keys()))
        rng.shuffle(drug_ids)

        # Select drugs one by one until the desired number of test samples is reached.
        test_indices: set = set()
        for d in drug_ids:
            test_indices.update(drug_to_indices[d])
            if len(test_indices) >= target_test:
                break

        test_idx = np.array(sorted(test_indices), dtype=np.int64)

        # Remaining samples for training/validation, split by ratio
        remaining_mask = np.ones(n_total, dtype=bool)
        remaining_mask[test_idx] = False
        remaining_indices = np.nonzero(remaining_mask)[0]

        rng_remaining = rng.permutation(len(remaining_indices)) if len(remaining_indices) > 0 else np.array([], dtype=np.int64)
        n_val = int(len(remaining_indices) * float(val_ratio))

        val_idx = remaining_indices[rng_remaining[:n_val]] if n_val > 0 else np.array([], dtype=np.int64)
        train_idx = remaining_indices[rng_remaining[n_val:]] if len(remaining_indices) > 0 else np.array([], dtype=np.int64)

        return train_idx, val_idx, test_idx

    @staticmethod
    def split_data_generalization(triples: np.ndarray, val_ratio: float = 0.1,
                                  test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Based on a dataset categorized by drug entities, we ensure that some drugs in the test set do not appear in the training/validation sets.
        """
        train_idx, val_idx, test_idx = DataSplittingModule._entity_holdout_split_indices(
            triples, val_ratio, test_ratio, random_seed
        )

        train_data = triples[train_idx]
        val_data = triples[val_idx]
        test_data = triples[test_idx]

        # Count the drug coverage to confirm no overlap
        train_drugs = set(np.unique(train_data[:, :2].reshape(-1))) if len(train_data) > 0 else set()
        val_drugs = set(np.unique(val_data[:, :2].reshape(-1))) if len(val_data) > 0 else set()
        test_drugs = set(np.unique(test_data[:, :2].reshape(-1))) if len(test_data) > 0 else set()

        overlap_train_test = len(train_drugs.intersection(test_drugs))
        overlap_val_test = len(val_drugs.intersection(test_drugs))

        print(f"[general] 训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")

        return train_data, val_data, test_data

    @staticmethod
    def split_multilabel_data(triples: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1,
                             test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split multi-label dataset.

        Args:
            triples: Triple data.
            labels: Label data.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            random_seed: Random seed.

        Returns:
            Tuple containing:
            - train_triples, train_labels: Training set triples and labels
            - val_triples, val_labels: Validation set triples and labels
            - test_triples, test_labels: Test set triples and labels
        """
        rng = np.random.RandomState(random_seed)
        idx = rng.permutation(len(triples))

        n_total = len(triples)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test+n_val], idx[n_test+n_val:]

        train_triples, train_labels = triples[train_idx], labels[train_idx]
        val_triples, val_labels = triples[val_idx], labels[val_idx]
        test_triples, test_labels = triples[test_idx], labels[test_idx]

        return train_triples, train_labels, val_triples, val_labels, test_triples, test_labels

    @staticmethod
    def split_multilabel_data_generalization(triples: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1,
                                            test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split multi-label dataset (drug entity split) to ensure some drugs in the test set do not appear in the training/validation sets.
        """
        train_idx, val_idx, test_idx = DataSplittingModule._entity_holdout_split_indices(
            triples, val_ratio, test_ratio, random_seed
        )

        train_triples, train_labels = triples[train_idx], labels[train_idx]
        val_triples, val_labels = triples[val_idx], labels[val_idx]
        test_triples, test_labels = triples[test_idx], labels[test_idx]

        train_drugs = set(np.unique(train_triples[:, :2].reshape(-1))) if len(train_triples) > 0 else set()
        val_drugs = set(np.unique(val_triples[:, :2].reshape(-1))) if len(val_triples) > 0 else set()
        test_drugs = set(np.unique(test_triples[:, :2].reshape(-1))) if len(test_triples) > 0 else set()

        print(f"[general] multilabel 训练集: {train_triples.shape}, 验证集: {val_triples.shape}, 测试集: {test_triples.shape}")
        print(f"[general] multilabel 训练药物数: {len(train_drugs)}, 验证药物数: {len(val_drugs)}, 测试药物数: {len(test_drugs)}")
        print(f"[general] multilabel 训练-测试药物交集: {len(train_drugs.intersection(test_drugs))}, 验证-测试药物交集: {len(val_drugs.intersection(test_drugs))}")

        return train_triples, train_labels, val_triples, val_labels, test_triples, test_labels

    @staticmethod
    def add_label_noise_multiclass(train_data: np.ndarray, num_classes: int,
                                  noise_ratio: float, random_seed: int = 1) -> np.ndarray:
        """
        Inject label noise for multi-class classification.

        Args:
            train_data: Training data.
            num_classes: Number of classes.
            noise_ratio: Noise ratio.
            random_seed: Random seed.

        Returns:
            Training data with added label noise.
        """
        if noise_ratio <= 0:
            return train_data

        rng = np.random.RandomState(random_seed)
        n_train = len(train_data)
        flip_n = int(n_train * float(noise_ratio))
        idx_sel = rng.choice(n_train, size=flip_n, replace=False)

        for ii in idx_sel:
            y0 = train_data[ii, 2]
            cand = [c for c in range(num_classes) if c != y0]
            train_data[ii, 2] = rng.choice(cand)

        print(f"训练集标签噪声比例: {noise_ratio}, 影响样本数: {len(idx_sel)}")
        return train_data

    @staticmethod
    def add_label_noise_multilabel(labels: np.ndarray, noise_ratio: float,
                                  flip_per_label: int = 50, random_seed: int = 1) -> np.ndarray:
        """
        Inject label noise for multi-label classification.

        Args:
            labels: Label matrix.
            noise_ratio: Noise ratio.
            flip_per_label: Number of flipped labels per sample.
            random_seed: Random seed.

        Returns:
            Label matrix with added noise.
        """
        if noise_ratio <= 0:
            return labels

        rng = np.random.RandomState(random_seed)
        n_train = len(labels)
        flip_n = int(n_train * float(noise_ratio))
        idx_sel = rng.choice(n_train, size=flip_n, replace=False)
        num_classes = labels.shape[1]

        for ii in idx_sel:
            flip_indices = rng.choice(num_classes, size=flip_per_label, replace=False)
            labels[idx_sel[ii], flip_indices] = 1.0 - labels[idx_sel[ii], flip_indices]

        print(f"训练集标签噪声比例: {noise_ratio}, 影响样本数: {len(idx_sel)}, 每样本翻转标签数: {flip_per_label}")
        return labels

    @staticmethod
    def sparse_sampling_multiclass(train_data: np.ndarray, num_classes: int,
                                  sparse_sample_rate: float, random_seed: int = 1) -> np.ndarray:
        """
        Perform sparse sampling for multi-class classification.

        Args:
            train_data: Training data.
            num_classes: Number of classes.
            sparse_sample_rate: Sampling rate.
            random_seed: Random seed.

        Returns:
            Sampled training data.
        """
        if sparse_sample_rate <= 0:
            return train_data

        if not (0.0 < sparse_sample_rate < 1.0):
            raise ValueError("sparse_sample_rate 必须在 (0, 1) 范围内")

        rng = np.random.RandomState(random_seed)
        labels = train_data[:, 2]
        n_train = len(train_data)

        # Calculate frequency of each label
        label_counts = np.bincount(labels.astype(int), minlength=num_classes)
        print(f"采样前各标签频率: {label_counts}")

        # Calculate number of samples to keep for each label
        keep_ratios = 1.0 - sparse_sample_rate
        keep_counts = (label_counts * keep_ratios).astype(int)
        keep_counts = np.maximum(keep_counts, 1)

        # Initialize list of indices to keep
        keep_indices = []

        # Sample for each label
        for label in range(num_classes):
            label_indices = np.where(labels == label)[0]
            n_keep = keep_counts[label]
            if n_keep >= len(label_indices):
                keep_indices.extend(label_indices)
            else:
                selected_indices = rng.choice(label_indices, size=n_keep, replace=False)
                keep_indices.extend(selected_indices)

        train_data = train_data[keep_indices]
        print(f"采样后训练集大小: {train_data.shape}")
        print(f"采样后各标签频率: {np.bincount(train_data[:, 2].astype(int), minlength=num_classes)}")

        return train_data


# =============================================================================
# Graph Construction Module
# =============================================================================

class GraphConstructionModule:
    """
    Graph construction module.
    
    Responsible for building graph structures from training data, 
    supporting multi-relation and single-relation graphs.
    """

    @staticmethod
    def build_multigraph(train_data: np.ndarray, network_ratio: float = 1.0,
                        random_seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build multi-relation graph.

        Args:
            train_data: Training data.
            network_ratio: Edge usage ratio.
            random_seed: Random seed.

        Returns:
            Tuple containing:
            - edge_index: Edge index tensor
            - edge_type: Edge type tensor
        """
        use_ratio = float(network_ratio)
        if use_ratio <= 0 or use_ratio > 1:
            use_ratio = 1.0

        edges = train_data
        if use_ratio < 1.0:
            keep = int(max(1, round(edges.shape[0] * use_ratio)))
            sel = np.random.RandomState(random_seed).permutation(edges.shape[0])[:keep]
            edges = edges[sel]
            print(f"[graph] using edge_ratio={use_ratio} -> {keep}/{train_data.shape[0]} edges for RGCN graph.")
        else:
            print(f"[graph] using all {train_data.shape[0]} edges for RGCN graph.")

        edge_index, edge_type = [], []
        for i, j, r in edges:
            i = int(i); j = int(j); r = int(r)
            edge_index.append([i, j]); edge_type.append(r)
            edge_index.append([j, i]); edge_type.append(r)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        return edge_index, edge_type

    @staticmethod
    def build_single_relation_graph(train_triples: np.ndarray, network_ratio: float = 1.0,
                                   random_seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build single-relation graph.

        Args:
            train_triples: Training triples.
            network_ratio: Edge usage ratio.
            random_seed: Random seed.

        Returns:
            Tuple containing:
            - edge_index: Edge index tensor
            - edge_type: Edge type tensor (all zeros)
        """
        use_ratio = float(network_ratio)
        if use_ratio <= 0 or use_ratio > 1:
            use_ratio = 1.0

        edges = train_triples
        if use_ratio < 1.0:
            keep = int(max(1, round(edges.shape[0] * use_ratio)))
            sel = np.random.RandomState(random_seed).permutation(edges.shape[0])[:keep]
            edges = edges[sel]
            print(f"[graph] using edge_ratio={use_ratio} -> {keep}/{len(train_triples)} edges for RGCN graph.")
        else:
            print(f"[graph] using all {len(train_triples)} edges for RGCN graph.")

        edge_index, edge_type = [], []
        for i, j, _ in edges:
            i = int(i); j = int(j); r = 0
            edge_index.append([i, j]); edge_type.append(r)
            edge_index.append([j, i]); edge_type.append(r)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        return edge_index, edge_type


# =============================================================================
# DataLoader Creation Module
# =============================================================================

class DataLoaderCreationModule:
    """
    DataLoader creation module.
    
    Responsible for creating and configuring DataLoaders.
    """

    @staticmethod
    def create_dataloader_config(args: argparse.Namespace) -> Dict:
        """
        Create DataLoader configuration.

        Args:
            args: Configuration parameters.

        Returns:
            DataLoader configuration dictionary.
        """
        params = {
            'batch_size': args.batch,
            'shuffle': False,
            'num_workers': int(getattr(args, 'workers', 0)),
            'drop_last': False,
            'pin_memory': False,
            'persistent_workers': False,
        }

        if params['num_workers'] > 0:
            params['prefetch_factor'] = 1

        return params

    @staticmethod
    def create_multiclass_dataloaders(train_data: np.ndarray, val_data: np.ndarray,
                                      test_data: np.ndarray, args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create multi-class DataLoaders.

        Args:
            train_data: Training data.
            val_data: Validation data.
            test_data: Test data.
            args: Configuration parameters.

        Returns:
            Tuple containing:
            - train_loader: Training DataLoader
            - val_loader: Validation DataLoader
            - test_loader: Test DataLoader
        """
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseMultiDataset(train_data), **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseMultiDataset(val_data), **params)
        test_loader = DataLoader(BaseMultiDataset(test_data), **params)

        return train_loader, val_loader, test_loader

    @staticmethod
    def create_multilabel_dataloaders(train_triples: np.ndarray, train_labels: np.ndarray,
                                     val_triples: np.ndarray, val_labels: np.ndarray,
                                     test_triples: np.ndarray, test_labels: np.ndarray,
                                     args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create multi-label DataLoaders.

        Args:
            train_triples: Training triples.
            train_labels: Training labels.
            val_triples: Validation triples.
            val_labels: Validation labels.
            test_triples: Test triples.
            test_labels: Test labels.
            args: Configuration parameters.

        Returns:
            Tuple containing:
            - train_loader: Training DataLoader
            - val_loader: Validation DataLoader
            - test_loader: Test DataLoader
        """
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseMultiLabelDataset(train_triples, train_labels), **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseMultiLabelDataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BaseMultiLabelDataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


# =============================================================================
# Base Dataset Classes
# =============================================================================

class BaseMultiDataset(Dataset):
    """Multi-class dataset class."""
    def __init__(self, triple: np.ndarray):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype = triple[:, 2]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):
        return (self.entity1[index], self.entity2[index], self.relationtype[index])


class BaseMultiLabelDataset(Dataset):
    """Multi-label dataset class."""
    def __init__(self, triple: np.ndarray, labels: np.ndarray = None):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.labels = labels
        self.relationtype = triple[:, 2] if labels is None else np.zeros(len(triple), dtype=np.int64)

    def __len__(self):
        return len(self.entity1)

    def __getitem__(self, index):
        if self.labels is not None:
            return (self.entity1[index], self.entity2[index], self.labels[index])
        return (self.entity1[index], self.entity2[index], 0)


# =============================================================================
# Main BaseDataset Class
# =============================================================================

class BaseDataset:
    """
    Base dataset class integrating all functional modules.
    
    Provides a unified data processing interface supporting 
    multi-class and multi-label tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_o: Data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Initialize functional modules
        self.data_loader = DataLoadingModule()
        self.feature_processor = FeatureProcessingModule()
        self.data_splitter = DataSplittingModule()
        self.graph_builder = GraphConstructionModule()
        self.dataloader_creator = DataLoaderCreationModule()

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Main entry point for loading data.

        Args:
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
        """
        if self.args.matrix in ['multilabel', 'twosides']:
            self._load_multilabel_data(val_ratio, test_ratio)
        else:
            self._load_multiclass_data(val_ratio, test_ratio)

    def _load_multiclass_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """Load multi-class data."""
        print("=== 开始加载多分类数据 ===")


        # 1. Read node embeddings
        id2vec, emb_dim = self.data_loader.read_id_embedding_pt(self.args.embedding_path)
        print(f"嵌入维数: {emb_dim}")

        # 2. Read pairs and remap
        pairs_df, num_relations = self.data_loader.read_multi_pairs_and_remap(self.args.matrix_path)
        print(f"DDI 类型数量: {num_relations}")
        self.args.num_classes = int(num_relations)

        # 3. Build feature matrix
        drug_list = sorted(set(pairs_df['id1']).union(set(pairs_df['id2'])))
        x = self.feature_processor.build_feature_matrix(drug_list, id2vec, emb_dim, self.args)
        self.args.dimensions = int(x.shape[1])

        # 4. Build triples
        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        triples = np.asarray(
            [(drug_id_to_index[h], drug_id_to_index[t], int(r))
             for h, t, r in zip(pairs_df['id1'], pairs_df['id2'], pairs_df['ddi'])],
            dtype=np.int64
        )

        # 5. data splitting
        if getattr(self.args, 'general', True):
            train_data, val_data, test_data = self.data_splitter.split_data_generalization(
                triples, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )
        else:
            train_data, val_data, test_data = self.data_splitter.split_data(
                triples, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )

        # 6. Label noise processing
        if getattr(self.args, 'noise_ratio', 0.0) > 0:
            train_data = self.data_splitter.add_label_noise_multiclass(
                train_data, self.args.num_classes, self.args.noise_ratio
            )

        # 7. Sparse sampling
        if getattr(self.args, 'sparse_sample_rate', 0.0) > 0:
            train_data = self.data_splitter.sparse_sampling_multiclass(
                train_data, self.args.num_classes, self.args.sparse_sample_rate
            )

        # 8. Create DataLoader
        self.train_loader, self.val_loader, self.test_loader = self.dataloader_creator.create_multiclass_dataloaders(
            train_data, val_data, test_data, self.args
        )

        # 9. Build graph
        edge_index, edge_type = self.graph_builder.build_multigraph(
            train_data, getattr(self.args, 'network_ratio', 1.0)
        )
        self.data_o = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        print("=== 多分类数据加载完成 ===")

    def _load_multilabel_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """Load multi-label data."""
        print("=== 开始加载多标签数据 ===")


        # 1. Read node embeddings
        id2vec, emb_dim = self.data_loader.read_id_embedding_pt(self.args.embedding_path)
        print(f"嵌入维数: {emb_dim}")

        # 2. Read pairs and remap
        pairs_df, num_relations = self.data_loader.read_multilabel_pairs_and_remap(self.args.matrix_path)
        print(f"DDI 类型数量: {num_relations}")
        self.args.num_classes = int(num_relations)

        # 3. Build feature matrix
        drug_list = sorted(set(pairs_df['id1']).union(set(pairs_df['id2'])))
        x = self.feature_processor.build_feature_matrix(drug_list, id2vec, emb_dim, self.args)
        self.args.dimensions = int(x.shape[1])

        # 4. Build triples and labels
        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        triples = np.asarray(
            [(drug_id_to_index[h], drug_id_to_index[t], 0)
             for h, t in zip(pairs_df['id1'], pairs_df['id2'])],
            dtype=np.int64
        )
        labels = np.stack(pairs_df['ddi'].values).astype(np.float32)

        # 5. data splitting
        if getattr(self.args, 'general', True):
            train_triples, train_labels, val_triples, val_labels, test_triples, test_labels = self.data_splitter.split_multilabel_data_generalization(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )
        else:
            train_triples, train_labels, val_triples, val_labels, test_triples, test_labels = self.data_splitter.split_multilabel_data(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )

        # 6. Label noise processing
        if getattr(self.args, 'noise_ratio', 0.0) > 0:
            train_labels = self.data_splitter.add_label_noise_multilabel(
                train_labels, self.args.noise_ratio, getattr(self.args, 'flip_per_label', 50)
            )

        # 7. Create DataLoader
        self.train_loader, self.val_loader, self.test_loader = self.dataloader_creator.create_multilabel_dataloaders(
            train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, self.args
        )

        # 8. Build single-relation graph
        edge_index, edge_type = self.graph_builder.build_single_relation_graph(
            train_triples, getattr(self.args, 'network_ratio', 1.0)
        )
        self.data_o = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        print("=== 多标签数据加载完成 ===")

    def get_data_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_nodes': self.data_o.x.shape[0] if self.data_o is not None else 0,
            'num_edges': self.data_o.edge_index.shape[1] if self.data_o is not None else 0,
            'feature_dim': self.args.dimensions if hasattr(self.args, 'dimensions') else 0,
            'num_classes': self.args.num_classes if hasattr(self.args, 'num_classes') else 0,
            'train_size': len(self.train_loader.dataset) if self.train_loader else 0,
            'val_size': len(self.val_loader.dataset) if self.val_loader else 0,
            'test_size': len(self.test_loader.dataset) if self.test_loader else 0,
        }
        return stats

    # -------------------------------------------------------------------------
    # TIGER reuse: Returns the original ID pairs and label assignments, used to directly construct x and y.
    # -------------------------------------------------------------------------
    def build_pairs_labels_splits(self, val_ratio: float = 0.1, test_ratio: float = 0.2,
                                  random_seed: Optional[int] = None,
                                  return_original_ids: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load paired data from the given args.matrix or matrix_path, and perform a random split in the same style as BaseDataset.
        Return (pairs, labels) for train/val/test splits.

        - For multi-class tasks: labels are int64 of shape [N]
        - For multi-label tasks: labels are float32 of shape [N, C]

        Args:
            val_ratio (float): Validation set ratio.
            test_ratio (float): Test set ratio.
            random_seed (int, optional): Random seed (default from args.seed or 1).
            return_original_ids (bool, optional): Whether to return original ID (string) pairs. 
                If False, returns index pairs.

        Returns:
            dict: {
                'train': (pairs, labels),
                'val': (pairs, labels),
                'test': (pairs, labels)
            }
        """
        seed = int(getattr(self.args, 'seed', 1)) if random_seed is None else int(random_seed)

        # Read pairings and labels (select parsing method based on task type)
        if self.args.matrix in ['multilabel', 'twosides']:
            pairs_df, num_relations = self.data_loader.read_multilabel_pairs_and_remap(self.args.matrix_path)
            labels_all = np.stack(pairs_df['ddi'].values).astype(np.float32)
        else:
            pairs_df, num_relations = self.data_loader.read_multi_pairs_and_remap(self.args.matrix_path)
            labels_all = pairs_df['ddi'].to_numpy(dtype=np.int64)

        # Original ID pairs (strings)
        pairs_orig = pairs_df[['id1', 'id2']].to_numpy(dtype=object)

        # If not returning original IDs, remap based on drug_list
        if not return_original_ids:
            drug_list = sorted(set(pairs_df['id1']).union(set(pairs_df['id2'])))
            drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
            pairs_idx = np.asarray([(drug_id_to_index[h], drug_id_to_index[t])
                                    for h, t in pairs_orig], dtype=np.int64)
        else:
            pairs_idx = None  # Not used

        # Generate shuffled indices and split
        rng = np.random.RandomState(seed)
        n_total = len(pairs_df)
        perm = rng.permutation(n_total)
        n_test = int(n_total * float(test_ratio))
        n_val = int(n_total * float(val_ratio))
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        # Assemble output
        if return_original_ids:
            train_pairs = pairs_orig[train_idx]
            val_pairs = pairs_orig[val_idx]
            test_pairs = pairs_orig[test_idx]
        else:
            train_pairs = pairs_idx[train_idx]
            val_pairs = pairs_idx[val_idx]
            test_pairs = pairs_idx[test_idx]

        train_labels = labels_all[train_idx]
        val_labels = labels_all[val_idx]
        test_labels = labels_all[test_idx]

        return {
            'train': (train_pairs, train_labels),
            'val': (val_pairs, val_labels),
            'test': (test_pairs, test_labels),
        }
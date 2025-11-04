import os
import re
import pandas as pd
import numpy as np
import torch
import random
import argparse
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


# =============================================================================
# 数据加载和预处理模块 (Data Loading and Preprocessing Module)
# =============================================================================

class DataLoadingModule:
    """
    数据加载和预处理模块
    负责读取嵌入文件、配对数据文件以及基本的数据预处理
    """

    @staticmethod
    def read_id_embedding_pt(embedding_paths: Union[str, List[str]]) -> Tuple[Dict[str, np.ndarray], int]:
        """
        读取单个或多个模态的嵌入文件并将嵌入向量连接起来。

        Args:
            embedding_paths: 嵌入文件路径或路径列表

        Returns:
            id2vec: 映射 ID 到连接后的嵌入向量（numpy 数组）
            dim: 连接后嵌入向量的总维度
        """
        if isinstance(embedding_paths, str):
            embedding_paths = [embedding_paths]

        id2vec = {}
        total_dim = 0

        # 遍历所有模态的嵌入路径
        for path in embedding_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"未找到嵌入文件: {path}")

            # 读取嵌入文件
            data = torch.load(path)
            if not isinstance(data, dict):
                raise ValueError(f"期望 pt 文件中是 dict，但得到的是 {type(data)}")

            # 转换为 numpy 数组，保证 float32
            current_id2vec = {str(k): v.detach().cpu().numpy().astype(np.float32) for k, v in data.items()}

            # 获取当前模态的维度
            example_key = next(iter(current_id2vec))
            current_dim = current_id2vec[example_key].shape[0]
            total_dim += current_dim

            # 初始化 id2vec 或连接嵌入
            if not id2vec:
                id2vec = current_id2vec
            else:
                # 确保 ID 一致
                if set(id2vec.keys()) != set(current_id2vec.keys()):
                    raise ValueError(f"嵌入文件 {path} 的 ID 集与之前的嵌入不一致")
                # 连接每个 ID 的嵌入向量
                for id_ in id2vec:
                    id2vec[id_] = np.concatenate([id2vec[id_], current_id2vec[id_]], axis=0)

        return id2vec, total_dim

    @staticmethod
    def read_id_embedding_pt_split(embedding_paths: List[str]) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
        """
        读取多个模态的嵌入文件并分别存储嵌入向量。

        Args:
            embedding_paths: 嵌入文件路径列表

        Returns:
            modal2id2vec: 映射模态索引到该模态的 id2vec 字典
            dims: 每个模态的嵌入向量维度列表
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

        # 验证所有模态的 ID 集是否一致
        id_sets = [set(id2vec.keys()) for id2vec in modal2id2vec.values()]
        if len(set(frozenset(id_set) for id_set in id_sets)) > 1:
            raise ValueError("不同模态的嵌入文件中 ID 集不一致")

        return modal2id2vec, dims

    @staticmethod
    def read_multi_pairs_and_remap(matrix_path: str) -> Tuple[pd.DataFrame, int]:
        """
        读取多分类配对数据并进行标签重映射

        Args:
            matrix_path: 多分类数据文件路径

        Returns:
            df: 处理后的数据框，包含 id1, id2, ddi 列
            num_relations: 关系类型数量
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

        # 按唯一值重映射为连续标签
        unique_raw = np.sort(df2['ddi_raw'].unique())
        raw2new = {raw: i for i, raw in enumerate(unique_raw)}
        df2['ddi'] = df2['ddi_raw'].map(raw2new).astype(int)
        num_relations = len(unique_raw)

        return df2[['id1','id2','ddi']], num_relations

    @staticmethod
    def read_multilabel_pairs_and_remap(matrix_path: str) -> Tuple[pd.DataFrame, int]:
        """
        读取多标签配对数据

        Args:
            matrix_path: 多标签数据文件路径

        Returns:
            df: 处理后的数据框，包含 id1, id2, ddi 列
            num_ddi: 标签数量
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
# 特征处理和噪声注入模块 (Feature Processing and Noise Injection Module)
# =============================================================================

class FeatureProcessingModule:
    """
    特征处理和噪声注入模块
    负责特征矩阵的构建、标准化、噪声注入等处理
    """

    @staticmethod
    def build_feature_matrix(drug_list: List[str], id2vec: Dict[str, np.ndarray],
                           emb_dim: int, args: argparse.Namespace) -> torch.Tensor:
        """
        构建药物特征矩阵

        Args:
            drug_list: 药物ID列表
            id2vec: ID到嵌入向量的映射
            emb_dim: 嵌入维度
            args: 参数配置

        Returns:
            x: 标准化后的特征矩阵
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

        # 特征标准化
        feats = FeatureProcessingModule.normalize_features(feats)

        # 添加高斯噪声
        if getattr(args, 'noise_std', 0.0) > 0:
            feats = FeatureProcessingModule.add_gaussian_noise(feats, args.noise_std)

        # 稀疏丢弃
        if float(getattr(args, 'sparse_drop_rate', 0.0)) > 0:
            feats = FeatureProcessingModule.sparse_dropout(feats, args.sparse_drop_rate)

        return torch.tensor(feats, dtype=torch.float32)

    @staticmethod
    def normalize_features(feats: np.ndarray) -> np.ndarray:
        """
        特征标准化

        Args:
            feats: 原始特征矩阵

        Returns:
            标准化后的特征矩阵
        """
        return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    @staticmethod
    def add_gaussian_noise(feats: np.ndarray, noise_std: float) -> np.ndarray:
        """
        添加高斯噪声

        Args:
            feats: 原始特征矩阵
            noise_std: 噪声标准差

        Returns:
            添加噪声后的特征矩阵
        """
        noise = np.random.normal(0, float(noise_std), feats.shape).astype(np.float32)
        return feats + noise

    @staticmethod
    def sparse_dropout(feats: np.ndarray, drop_rate: float) -> np.ndarray:
        """
        稀疏丢弃

        Args:
            feats: 原始特征矩阵
            drop_rate: 丢弃比例

        Returns:
            稀疏化后的特征矩阵
        """
        mask = (np.random.rand(*feats.shape) > drop_rate).astype(np.float32)
        return feats * mask


# =============================================================================
# 数据分割和采样模块 (Data Splitting and Sampling Module)
# =============================================================================

class DataSplittingModule:
    """
    数据分割和采样模块
    负责数据集的划分、标签噪声注入、稀疏采样等
    """

    @staticmethod
    def split_data(triples: np.ndarray, val_ratio: float = 0.1,
                  test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数据集划分

        Args:
            triples: 三元组数据
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            train_data: 训练集
            val_data: 验证集
            test_data: 测试集
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
    def split_multilabel_data(triples: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1,
                             test_ratio: float = 0.2, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        多标签数据集划分

        Args:
            triples: 三元组数据
            labels: 标签数据
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            train_triples, train_labels: 训练集三元组和标签
            val_triples, val_labels: 验证集三元组和标签
            test_triples, test_labels: 测试集三元组和标签
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
    def add_label_noise_multiclass(train_data: np.ndarray, num_classes: int,
                                  noise_ratio: float, random_seed: int = 1) -> np.ndarray:
        """
        多分类标签噪声注入

        Args:
            train_data: 训练数据
            num_classes: 类别数量
            noise_ratio: 噪声比例
            random_seed: 随机种子

        Returns:
            添加噪声后的训练数据
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
        多标签标签噪声注入

        Args:
            labels: 标签矩阵
            noise_ratio: 噪声比例
            flip_per_label: 每个样本翻转的标签位数
            random_seed: 随机种子

        Returns:
            添加噪声后的标签矩阵
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
        多分类稀疏采样

        Args:
            train_data: 训练数据
            num_classes: 类别数量
            sparse_sample_rate: 采样率
            random_seed: 随机种子

        Returns:
            采样后的训练数据
        """
        if sparse_sample_rate <= 0:
            return train_data

        if not (0.0 < sparse_sample_rate < 1.0):
            raise ValueError("sparse_sample_rate 必须在 (0, 1) 范围内")

        rng = np.random.RandomState(random_seed)
        labels = train_data[:, 2]
        n_train = len(train_data)

        # 计算每个标签的频率
        label_counts = np.bincount(labels.astype(int), minlength=num_classes)
        print(f"采样前各标签频率: {label_counts}")

        # 计算每个标签需要保留的样本数
        keep_ratios = 1.0 - sparse_sample_rate
        keep_counts = (label_counts * keep_ratios).astype(int)
        keep_counts = np.maximum(keep_counts, 1)

        # 初始化保留样本的索引列表
        keep_indices = []

        # 对每个标签进行采样
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
# 图构建模块 (Graph Construction Module)
# =============================================================================

class GraphConstructionModule:
    """
    图构建模块
    负责根据训练数据构建图结构，支持多关系和单关系图
    """

    @staticmethod
    def build_multigraph(train_data: np.ndarray, network_ratio: float = 1.0,
                        random_seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建多关系图

        Args:
            train_data: 训练数据
            network_ratio: 边使用比例
            random_seed: 随机种子

        Returns:
            edge_index: 边索引张量
            edge_type: 边类型张量
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
        构建单关系图

        Args:
            train_triples: 训练三元组
            network_ratio: 边使用比例
            random_seed: 随机种子

        Returns:
            edge_index: 边索引张量
            edge_type: 边类型张量（全为0）
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
# DataLoader创建模块 (DataLoader Creation Module)
# =============================================================================

class DataLoaderCreationModule:
    """
    DataLoader创建模块
    负责创建和配置DataLoader
    """

    @staticmethod
    def create_dataloader_config(args: argparse.Namespace) -> Dict:
        """
        创建DataLoader配置

        Args:
            args: 参数配置

        Returns:
            DataLoader配置字典
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
        创建多分类DataLoader

        Args:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            args: 参数配置

        Returns:
            train_loader: 训练DataLoader
            val_loader: 验证DataLoader
            test_loader: 测试DataLoader
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
        创建多标签DataLoader

        Args:
            train_triples, train_labels: 训练三元组和标签
            val_triples, val_labels: 验证三元组和标签
            test_triples, test_labels: 测试三元组和标签
            args: 参数配置

        Returns:
            train_loader: 训练DataLoader
            val_loader: 验证DataLoader
            test_loader: 测试DataLoader
        """
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseMultiLabelDataset(train_triples, train_labels), **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseMultiLabelDataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BaseMultiLabelDataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


# =============================================================================
# 基础数据集类 (Base Dataset Classes)
# =============================================================================

class BaseMultiDataset(Dataset):
    """多分类数据集类"""
    def __init__(self, triple: np.ndarray):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype = triple[:, 2]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):
        return (self.entity1[index], self.entity2[index], self.relationtype[index])


class BaseMultiLabelDataset(Dataset):
    """多标签数据集类"""
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
# 主要BaseDataset类 (Main BaseDataset Class)
# =============================================================================

class BaseDataset:
    """
    基础数据集类，整合所有功能模块
    提供统一的数据处理接口，支持多分类和多标签任务
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_o: Data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # 初始化各个功能模块
        self.data_loader = DataLoadingModule()
        self.feature_processor = FeatureProcessingModule()
        self.data_splitter = DataSplittingModule()
        self.graph_builder = GraphConstructionModule()
        self.dataloader_creator = DataLoaderCreationModule()

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        加载数据的主入口

        Args:
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        if self.args.matrix in ['multilabel', 'twosides']:
            self._load_multilabel_data(val_ratio, test_ratio)
        else:
            self._load_multiclass_data(val_ratio, test_ratio)

    def _load_multiclass_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """加载多分类数据"""
        print("=== 开始加载多分类数据 ===")

        # 1. 读取节点嵌入
        id2vec, emb_dim = self.data_loader.read_id_embedding_pt(self.args.embedding_path)
        print(f"嵌入维数: {emb_dim}")

        # 2. 读取配对与重映射
        pairs_df, num_relations = self.data_loader.read_multi_pairs_and_remap(self.args.matrix_path)
        print(f"DDI 类型数量: {num_relations}")
        self.args.num_classes = int(num_relations)

        # 3. 构建特征矩阵
        drug_list = sorted(set(pairs_df['id1']).union(set(pairs_df['id2'])))
        x = self.feature_processor.build_feature_matrix(drug_list, id2vec, emb_dim, self.args)
        self.args.dimensions = int(x.shape[1])

        # 4. 构建三元组
        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        triples = np.asarray(
            [(drug_id_to_index[h], drug_id_to_index[t], int(r))
             for h, t, r in zip(pairs_df['id1'], pairs_df['id2'], pairs_df['ddi'])],
            dtype=np.int64
        )

        # 5. 数据划分
        train_data, val_data, test_data = self.data_splitter.split_data(
            triples, val_ratio, test_ratio
        )

        # 6. 标签噪声处理
        if getattr(self.args, 'noise_ratio', 0.0) > 0:
            train_data = self.data_splitter.add_label_noise_multiclass(
                train_data, self.args.num_classes, self.args.noise_ratio
            )

        # 7. 稀疏采样
        if getattr(self.args, 'sparse_sample_rate', 0.0) > 0:
            train_data = self.data_splitter.sparse_sampling_multiclass(
                train_data, self.args.num_classes, self.args.sparse_sample_rate
            )

        # 8. 创建DataLoader
        self.train_loader, self.val_loader, self.test_loader = self.dataloader_creator.create_multiclass_dataloaders(
            train_data, val_data, test_data, self.args
        )

        # 9. 构建图
        edge_index, edge_type = self.graph_builder.build_multigraph(
            train_data, getattr(self.args, 'network_ratio', 1.0)
        )
        self.data_o = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        print("=== 多分类数据加载完成 ===")

    def _load_multilabel_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """加载多标签数据"""
        print("=== 开始加载多标签数据 ===")

        # 1. 读取节点嵌入
        id2vec, emb_dim = self.data_loader.read_id_embedding_pt(self.args.embedding_path)
        print(f"嵌入维数: {emb_dim}")

        # 2. 读取配对与重映射
        pairs_df, num_relations = self.data_loader.read_multilabel_pairs_and_remap(self.args.matrix_path)
        print(f"DDI 类型数量: {num_relations}")
        self.args.num_classes = int(num_relations)

        # 3. 构建特征矩阵
        drug_list = sorted(set(pairs_df['id1']).union(set(pairs_df['id2'])))
        x = self.feature_processor.build_feature_matrix(drug_list, id2vec, emb_dim, self.args)
        self.args.dimensions = int(x.shape[1])

        # 4. 构建三元组和标签
        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        triples = np.asarray(
            [(drug_id_to_index[h], drug_id_to_index[t], 0)
             for h, t in zip(pairs_df['id1'], pairs_df['id2'])],
            dtype=np.int64
        )
        labels = np.stack(pairs_df['ddi'].values).astype(np.float32)

        # 5. 数据划分
        train_triples, train_labels, val_triples, val_labels, test_triples, test_labels = self.data_splitter.split_multilabel_data(
            triples, labels, val_ratio, test_ratio
        )

        # 6. 标签噪声处理
        if getattr(self.args, 'noise_ratio', 0.0) > 0:
            train_labels = self.data_splitter.add_label_noise_multilabel(
                train_labels, self.args.noise_ratio, getattr(self.args, 'flip_per_label', 50)
            )

        # 7. 创建DataLoader
        self.train_loader, self.val_loader, self.test_loader = self.dataloader_creator.create_multilabel_dataloaders(
            train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, self.args
        )

        # 8. 构建单关系图
        edge_index, edge_type = self.graph_builder.build_single_relation_graph(
            train_triples, getattr(self.args, 'network_ratio', 1.0)
        )
        self.data_o = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        print("=== 多标签数据加载完成 ===")

    def get_data_stats(self) -> Dict:
        """获取数据集统计信息"""
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
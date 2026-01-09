import argparse
import numpy as np
from torch_geometric.data import Data
from data.BaseDataset import BaseDataset

class UnifiedDataset(BaseDataset):
    """
    Unified数据集类，继承自BaseDataset

    功能特点:
    - 统一读取 id->embedding（可多模态拼接，见 args.embedding_path/embedding_dir + --modality）
    - 多分类：使用真实关系类型作为 edge_type（RGCN 会用到）
    - 多标签：图用单关系（edge_type=0）
    - 支持特征高斯噪声 (noise_std)，标签翻转噪声 (noise_ratio)
    - DataLoader：pin_memory=False, persistent_workers=False；workers>0 时 prefetch_factor=1
    - 支持稀疏采样 (sparse_sample_rate) 和稀疏丢弃 (sparse_drop_rate)
    """

    def __init__(self, args: argparse.Namespace):
        """
        初始化数据集

        Args:
            args: 包含以下参数的命名空间对象:
                - matrix: 数据类型 ('multilabel', 'twosides' 或其他多分类类型)
                - embedding_path: 嵌入文件路径
                - matrix_path: 矩阵数据文件路径
                - batch: 批大小
                - workers: 工作进程数 (可选)
                - noise_std: 特征高斯噪声标准差 (可选)
                - noise_ratio: 标签噪声比例 (可选)
                - sparse_sample_rate: 稀疏采样率 (可选)
                - sparse_drop_rate: 稀疏丢弃率 (可选)
                - network_ratio: 图边使用比例 (可选)
                - flip_per_label: 多标签翻转位数 (可选，默认50)
        """
        super().__init__(args)

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        加载数据的主入口方法

        Args:
            val_ratio: 验证集比例，默认0.1
            test_ratio: 测试集比例，默认0.2
        """
        print("=== Unified Dataset Loading ===")
        print(f"数据类型: {self.args.matrix}")
        print(f"嵌入路径: {self.args.embedding_path}")
        print(f"矩阵路径: {self.args.matrix_path}")

        # 调用父类的load_data方法，它会根据matrix类型自动选择加载方式
        super().load_data(val_ratio, test_ratio)

        # 打印数据集统计信息
        # self._print_dataset_info()

    def _print_dataset_info(self):
        """打印数据集统计信息"""
        stats = self.get_data_stats()
        print("\n=== 数据集统计信息 ===")
        print(f"节点数量: {stats['num_nodes']}")
        print(f"边数量: {stats['num_edges']}")
        print(f"特征维度: {stats['feature_dim']}")
        print(f"类别数量: {stats['num_classes']}")
        print(f"训练集大小: {stats['train_size']}")
        print(f"验证集大小: {stats['val_size']}")
        print(f"测试集大小: {stats['test_size']}")
        print("========================\n")

    def get_noise_config(self) -> dict:
        """
        获取噪声配置信息

        Returns:
            dict: 包含噪声配置的字典
        """
        noise_config = {
            'feature_noise_std': getattr(self.args, 'noise_std', 0.0),
            'label_noise_ratio': getattr(self.args, 'noise_ratio', 0.0),
            'sparse_drop_rate': getattr(self.args, 'sparse_drop_rate', 0.0),
            'sparse_sample_rate': getattr(self.args, 'sparse_sample_rate', 0.0),
            'network_ratio': getattr(self.args, 'network_ratio', 1.0),
        }

        if self.args.matrix in ['multilabel', 'twosides']:
            noise_config['flip_per_label'] = getattr(self.args, 'flip_per_label', 50)

        return noise_config


# 为了保持向后兼容性，保留原有的类名
class Unified_dataset(UnifiedDataset):
    """
    Unified_dataset类的别名，保持向后兼容性
    """
    pass
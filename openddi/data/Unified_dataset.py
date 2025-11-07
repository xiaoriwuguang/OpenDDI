import argparse
import numpy as np
from torch_geometric.data import Data
from data.BaseDataset import BaseDataset

class UnifiedDataset(BaseDataset):
    """
    Unified dataset class that inherits from BaseDataset.
    
    Features:
    - Unified reading of id->embedding (supports multi-modal concatenation, 
      see args.embedding_path/embedding_dir + --modality)
    - Multi-class: Uses real relationship types as edge_type (used by RGCN)
    - Multi-label: Uses single relationship graph (edge_type=0)
    - Supports feature Gaussian noise (noise_std) and label flip noise (noise_ratio)
    - DataLoader: pin_memory=False, persistent_workers=False; prefetch_factor=1 when workers>0
    - Supports sparse sampling (sparse_sample_rate) and sparse dropping (sparse_drop_rate)
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the Unified dataset.
        
        Args:
            args: Namespace object containing the following parameters:
                - matrix: Data type ('multilabel', 'twosides' or other multi-class types)
                - embedding_path: Embedding file path
                - matrix_path: Matrix data file path
                - batch: Batch size
                - workers: Number of worker processes (optional)
                - noise_std: Feature Gaussian noise standard deviation (optional)
                - noise_ratio: Label noise ratio (optional)
                - sparse_sample_rate: Sparse sampling rate (optional)
                - sparse_drop_rate: Sparse drop rate (optional)
                - network_ratio: Graph edge usage ratio (optional)
                - flip_per_label: Multi-label flip bits (optional, default 50)
        """
        super().__init__(args)

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Main entry method for loading data.
        
        Args:
            val_ratio: Validation set ratio, defaults to 0.1
            test_ratio: Test set ratio, defaults to 0.2
        """
        print("=== MUFFIN Dataset Loading ===")
        print(f"数据类型: {self.args.matrix}")
        print(f"嵌入路径: {self.args.embedding_path}")
        print(f"矩阵路径: {self.args.matrix_path}")

        # Call parent class's load_data method, which automatically selects 
        # loading method based on matrix type
        super().load_data(val_ratio, test_ratio)

        # Print dataset statistics
        # self._print_dataset_info()

    def _print_dataset_info(self):
        """Print dataset statistics."""
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
        Get noise configuration information.
        
        Returns:
            dict: Dictionary containing noise configuration
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


# For backward compatibility, keep the original class name
class Unified_dataset_dataset(UnifiedDataset):
    """
    Alias for UnifiedDataset class to maintain backward compatibility.
    """
    pass
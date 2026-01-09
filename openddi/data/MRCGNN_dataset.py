import os
import re
import pandas as pd
import numpy as np
import torch
import random
import argparse
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from data.BaseDataset import BaseDataset

class MRCGNN_dataset(BaseDataset):
    def __init__(self,
                 args:argparse.ArgumentParser):
        super().__init__(args)
        self.data_s = None
        self.data_a = None
        self.predict_loader = None
        self.index_to_drug_id = None

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        super().load_data(val_ratio, test_ratio)

        if self.args.matrix in ['multilabel', 'twosides']:
            self._load_multilabel_data_additional()
        else :
            self._load_multi_data_additional()

    def _load_multi_data_additional(self):
        """
        MRCGNN特有的多分类数据加载逻辑，包括对抗样本数据构建
        """
        # 从BaseDataset中获取训练数据用于构图
        # 使用已加载的self.data_o来构建额外的对比学习数据

        # 对抗节点, 在MRCGNN中用作对比学习，随机扰动边/节点特征
        features_o = self.data_o.x.detach().cpu().numpy()
        id_perm = np.random.permutation(features_o.shape[0])
        x_a = torch.tensor(features_o[id_perm], dtype=torch.float)

        # 获取药物列表
        num_drugs = features_o.shape[0]
        y_a = torch.cat((torch.ones(num_drugs, 1), torch.zeros(num_drugs, 1)), dim=1)

        # 获取边和边类型信息用于对比学习
        edge_index_o = self.data_o.edge_index
        edge_types = self.data_o.edge_type

        # 构建边类型对数据
        edge_types_pair = []
        for i in range(0, len(edge_types), 2):  # 每条双向边取一个
            r = edge_types[i].item()
            edge_types_pair.extend([r, r])
        edge_types_pair = torch.tensor(edge_types_pair, dtype=torch.int64)

        # 创建MRCGNN特有的数据对象
        self.data_s = Data(x=x_a, edge_index=edge_index_o, edge_type=edge_types)
        self.data_a = Data(x=self.data_o.x, y=y_a, edge_type=edge_types_pair)


    def _load_multilabel_data_additional(self):
        """
        MRCGNN特有的多标签数据加载逻辑，包括对抗样本数据构建
        """
        # 从BaseDataset中获取训练数据用于构图
        # 使用已加载的self.data_o来构建额外的对比学习数据

        # 对抗节点, 在MRCGNN中用作对比学习，随机扰动边/节点特征
        features_o = self.data_o.x.detach().cpu().numpy()
        id_perm = np.random.permutation(features_o.shape[0])
        x_a = torch.tensor(features_o[id_perm], dtype=torch.float)

        # 获取药物列表
        num_drugs = features_o.shape[0]
        y_a = torch.cat((torch.ones(num_drugs, 1), torch.zeros(num_drugs, 1)), dim=1)

        # 获取边和边类型信息用于对比学习
        edge_index_o = self.data_o.edge_index
        edge_types = self.data_o.edge_type

        # 对于多标签，edge_types都是0
        edge_types_pair = []
        for i in range(0, len(edge_types), 2):  # 每条双向边取一个
            edge_types_pair.extend([0, 0])
        edge_types_pair = torch.tensor(edge_types_pair, dtype=torch.int64)

        # 创建MRCGNN特有的数据对象
        self.data_s = Data(x=x_a, edge_index=edge_index_o, edge_type=edge_types)
        self.data_a = Data(x=self.data_o.x, y=y_a, edge_type=edge_types_pair)


        
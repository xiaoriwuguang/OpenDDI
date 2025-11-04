import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TIGER(nn.Module):
    """
    TIGER (实现版)：图编码 + 对表示 + MLP 分类
    构造签名与你现有 model_manager 保持一致：
      (feature, hidden1, hidden2, num_relations, num_classes, dropout)
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # ---- 图编码：两层 GAT ----
        heads1, heads2 = 4, 4
        self.gnn1 = GATConv(self.feature, self.hid1 // heads1, heads=heads1, concat=True)
        self.gnn2 = GATConv(self.hid1,   self.hid1 // heads2, heads=heads2, concat=True)

        # ---- 对表示：concat + abs diff + hadamard ----
        pair_in_dim = 4 * self.hid1
        hid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.num_classes)  # 直接输出 logits
        )

        # 可绑定图（dataset.data_graph），便于 Trainer 每步少传参
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    def bind_graph(self, data_graph):
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, x, edge_index):
        h = self.gnn1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)
        return h

    @staticmethod
    def _pair_features(h, i_idx, j_idx):
        hi = h[i_idx]
        hj = h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        """
        graph_or_none: torch_geometric.data.Data 或 None（若 None 则用已缓存的图）
        idx_batch: (i_idx, j_idx, y) —— 与 Base_multi_dataset / Base_multilabel_dataset 对齐
        返回 logits: [B, K]
        """
        if graph_or_none is not None:
            x, edge_index = graph_or_none.x, graph_or_none.edge_index
        else:
            x, edge_index = self.x_cache, self.edge_index_cache
        assert x is not None and edge_index is not None, "graph 未绑定或传入"

        h = self._encode_nodes(x, edge_index)
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=h.device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=h.device)
        pair_feat = self._pair_features(h, i_idx, j_idx)  # [B, 4*hid2]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits
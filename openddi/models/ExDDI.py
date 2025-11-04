import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint

class FeatureMask(nn.Module):
    """
    特征解释性层：为每个维度学习一个权重 (0-1)。
    在降维后的 hid1 空间上做掩码，更省显存。
    """
    def __init__(self, in_dim:int):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.mask)  # [N, d] * [d]

class ExDDI(nn.Module):
    """
    ExDDI baseline :
      - Linear 降维 + FeatureMask (hid1 维)
      - 单层 GraphSAGE(hid1→hid2)
      - checkpoint（无 use_reentrant 参数）
      - Pair 表征 → MLP → logits
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1    = int(hidden1)
        self.hid2    = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # # 高维特征降到 hid1
        # self.proj = nn.Linear(self.feature, self.hid1)
        # 低维空间做掩码
        self.explainer = FeatureMask(self.feature)

        # 单层 GraphSAGE（更省显存）
        self.gnn = SAGEConv(self.feature, self.hid1)

        # Pair → 分类器
        pair_in = 4 * self.hid1
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, self.hid2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.hid2),     nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.num_classes)
        )

        # 缓存图
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    def bind_graph(self, data_graph):
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, X, edge_index):
        # z = self.proj(X)                          # [N, hid1]
        z = self.explainer(X)                     # [N, hid1]
        # checkpoint（旧版 torch 无 use_reentrant 参数）
        # h = checkpoint(lambda a,b: self.gnn(a,b), z, edge_index)
        h = self.gnn(z,edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h                                   # [N, hid2]

    @staticmethod
    def _pair_feat(h, i_idx, j_idx):
        hi, hj = h[i_idx], h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        if graph_or_none is not None:
            X, edge_index = graph_or_none.x, graph_or_none.edge_index
        else:
            X, edge_index = self.x_cache, self.edge_index_cache
        assert X is not None and edge_index is not None, "graph 未绑定或传入"

        h = self._encode_nodes(X, edge_index)

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pf = self._pair_feat(h, i_idx, j_idx)
        logits = self.mlp(pf)
        return logits

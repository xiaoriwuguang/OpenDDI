import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.utils.checkpoint import checkpoint

class DDKG(nn.Module):
    """
    DDKG baseline :
      - 线性降维: feature → hid1
      - 两层 RGCNConv，使用 num_bases 降低多关系参数/激活
      - checkpoint (无 use_reentrant 参数)
      - Pair 特征 → MLP → logits
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # 先将高维节点特征降到 hid1，减少边消息维度
        self.pre = nn.Linear(self.feature, self.hid1)

        # RGCN 使用低秩分解（bases）
        nb = min(self.num_relations, 16)
        self.rgcn1 = RGCNConv(self.feature, self.hid1,
                              num_relations=self.num_relations,
                              num_bases=nb)
        self.rgcn2 = RGCNConv(self.hid1, self.hid2,
                              num_relations=self.num_relations,
                              num_bases=nb)

        # Pair 表征 → 分类
        pair_in_dim = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, mid),         nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.num_classes)
        )

        # 缓存图
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)
        self.register_buffer("edge_type_cache", None)

    def bind_graph(self, data_graph):
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index
        self.edge_type_cache  = getattr(data_graph, "edge_type", None)

    def _encode_nodes(self, x, edge_index, edge_type):
        # 线性降维
        x1 = self.pre(x)
        # checkpoint（旧版 torch 无 use_reentrant 参数）
        h = self.rgcn1(x, edge_index, edge_type)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.rgcn2(h, edge_index, edge_type)
        return h

    @staticmethod
    def _pair_features(h, i_idx, j_idx):
        hi, hj = h[i_idx], h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        if graph_or_none is not None:
            x = graph_or_none.x
            edge_index = graph_or_none.edge_index
            edge_type  = getattr(graph_or_none, "edge_type", None)
        else:
            x, edge_index, edge_type = self.x_cache, self.edge_index_cache, self.edge_type_cache

        assert x is not None and edge_index is not None, "graph 未绑定或传入"
        assert edge_type is not None, "DDKG 需要 edge_type（关系类型）"

        h = self._encode_nodes(x, edge_index, edge_type)

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_feat = self._pair_features(h, i_idx, j_idx)  # [B, 4*hid2]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits

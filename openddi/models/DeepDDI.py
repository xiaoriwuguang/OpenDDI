# models/DeepDDI.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDDI(nn.Module):
    """
    DeepDDI baseline:
      - 不进行图卷积，直接使用节点特征表 X
      - 对表示: [x_i || x_j || |x_i - x_j| || x_i * x_j]
      - MLP -> logits  (支持单/多标签)
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        pair_in = 4 * self.feature
        hid = max(self.hid1, 256)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.hid2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.num_classes)  # logits
        )

        # 缓存节点特征矩阵 X（来自 dataset.data_graph.x），接口对齐 TIGER/ZeroDDI
        self.register_buffer("x_cache", None)

    def bind_graph(self, data_graph):
        # 仅用到 X，不用 edge_index
        self.x_cache = data_graph.x

    @staticmethod
    def _pair_features(X, i_idx, j_idx):
        xi = X[i_idx]
        xj = X[j_idx]
        return torch.cat([xi, xj, torch.abs(xi - xj), xi * xj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        """
        graph_or_none: 可传 None（已 bind），或传入 Data（只取其 x）
        idx_batch: (i_idx, j_idx, y)
        """
        X = graph_or_none.x if (graph_or_none is not None) else self.x_cache
        assert X is not None, "DeepDDI 需要节点特征表 X，请在 trainer 中先 bind_graph(dataset.data_graph)"
        device = X.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_feat = self._pair_features(X, i_idx, j_idx)  # [B, 4*feature]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits

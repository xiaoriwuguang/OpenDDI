# models/MMDGDTI.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # 也可换 GATConv/GCNConv

class ModalAttnFusion(nn.Module):
    """
    多模态注意力融合：
      x_m -> proj_m(x_m) in R^d
      s_m = w_m^T x_m  (打分)
      α = softmax([s_m])；h = sum_m α_m * ReLU(proj_m(x_m))
    """
    def __init__(self, modal_dims, out_dim, dropout=0.0):
        super().__init__()
        self.modal_dims = list(map(int, modal_dims))
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)

        self.proj = nn.ModuleList([nn.Linear(d, self.out_dim) for d in self.modal_dims])
        self.scor = nn.ModuleList([nn.Linear(d, 1)           for d in self.modal_dims])

    def forward(self, x, splits):
        # x: [N, sum(d_m)]；splits: [0, d1, d1+d2, ..., sum]
        feats = []
        scores = []
        for (l, r), pj, sc in zip(zip(splits[:-1], splits[1:]), self.proj, self.scor):
            xm = x[:, l:r]                   # [N, d_m]
            hm = F.relu(pj(xm))              # [N, out]
            sm = sc(xm)                      # [N, 1]
            feats.append(hm); scores.append(sm)
        S = torch.cat(scores, dim=1)         # [N, M]
        A = torch.softmax(S, dim=1)          # [N, M]
        H = torch.stack(feats, dim=1)        # [N, M, out]
        h = torch.sum(A.unsqueeze(-1) * H, dim=1)  # [N, out]
        return F.dropout(h, p=self.dropout, training=self.training), A  # 返回注意力备查

class MMDGDTI(nn.Module):
    """
    MMDG-DTI baseline（适配 DDI）：
      模态注意力融合 -> GraphSAGE(两层) -> Pair 表征 -> MLP 分类
    构造签名与 model_manager 保持一致：
      (feature, hidden1, hidden2, num_relations, num_classes, dropout)
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1    = int(hidden1)
        self.hid2    = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # --- 模态切分（运行期可通过 set_modal_splits 注入）---
        self.modal_dims = [self.feature]  # 默认为单模态（整合后）
        self.register_buffer("splits", torch.tensor([0, self.feature], dtype=torch.long))

        # --- 多模态融合（注意力） ---
        self.fuse = ModalAttnFusion(self.modal_dims, out_dim=self.hid1, dropout=self.dropout)

        # --- 图编码（两层 SAGE，稳健简洁；可换成 GAT/GCN/RGCN） ---
        self.gnn1 = SAGEConv(self.hid1, self.hid2)
        self.gnn2 = SAGEConv(self.hid2, self.hid2)

        # --- Pair 头（对称）：[hi, hj, |hi-hj|, hi*hj] -> logits ---
        pair_in = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, mid),     nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.num_classes)
        )

        # 缓存图
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    # 在 Trainer 初始化里调用，告知各模态维度（与 CSV 拼接顺序一致）
    def set_modal_splits(self, split_str: str):
        if not split_str:  # 保持单模态
            return
        dims = [int(s) for s in split_str.split(',') if s.strip()]
        assert sum(dims) == self.feature, f"modal_splits 之和需等于 feature={self.feature}, got {sum(dims)}"
        self.modal_dims = dims
        # 重建融合层
        self.fuse = ModalAttnFusion(self.modal_dims, out_dim=self.hid1, dropout=self.dropout).to(next(self.parameters()).device)
        # 构建边界
        acc = [0]
        for d in self.modal_dims: acc.append(acc[-1] + d)
        self.splits = torch.tensor(acc, dtype=torch.long, device=next(self.parameters()).device)

    def bind_graph(self, data_graph):
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, X, edge_index):
        # 多模态注意力融合
        if self.splits.device != X.device:
            self.splits = self.splits.to(X.device)
        h0, _ = self.fuse(X, self.splits)             # [N, hid1]

        # 图编码（两层 SAGE）
        h = self.gnn1(h0, edge_index); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)                  # [N, hid2]
        return h

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

        h = self._encode_nodes(X, edge_index)  # [N, hid2]

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pf = self._pair_feat(h, i_idx, j_idx)       # [B, 4*hid2]
        logits = self.mlp(pf)                        # [B, K]
        return logits

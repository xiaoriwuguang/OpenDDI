import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

__all__ = ["SumGNN"]

class SumGNN(nn.Module):
    """
    两层 RGCN；先将高维多模态特征投影到较低维，再做消息传递。
    pair 表示 = concat(h1_i, h1_j, h2_i, h2_j) -> 全连接输出 logits
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3,
                 proj_dim:int=None, num_bases:int=None):
        super().__init__()
        self.feature = int(feature)          # 原始多模态维度（2860）
        self.hidden1 = int(hidden1)          # 第一层隐层
        self.hidden2 = int(hidden2)          # 第二层隐层
        self.num_rel = max(1, int(num_relations))  # 关系数（你的图边类型都是0，其实=1即可）
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # ---- 关键：输入降维（把 2860 -> proj_dim，默认取 hidden1 的 2~4 倍）----
        if proj_dim is None:
            proj_dim = max(128, min(512, self.hidden1 * 4))
        self.proj_dim = int(proj_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(self.feature, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ---- RGCN 参数低秩分解：大幅降低权重显存 ----
        if num_bases is None:
            num_bases = min(self.num_rel, 2)  # 你的图基本是单关系，2 就够了
        self.num_bases = int(num_bases)

        self.gnn1 = RGCNConv(self.feature, self.hidden1,
                             num_relations=self.num_rel, num_bases=self.num_bases)
        self.gnn2 = RGCNConv(self.hidden1, self.hidden2,
                             num_relations=self.num_rel, num_bases=self.num_bases)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

        # pair 表示：[h1_i, h1_j, h2_i, h2_j]
        self.fc = nn.Linear(2*self.hidden1 + 2*self.hidden2, self.num_classes)

    def drug_feat(self, emb):
        # 兼容旧接口；不使用
        self.drugfeat = emb

    def forward(self, data_o, idx_batch):
        x, edge_index, edge_type = data_o.x, data_o.edge_index, data_o.edge_type
        a_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=x.device)
        b_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=x.device)

        # 输入先降维 -> 大幅减少后续 edge gather 显存
        # h0 = self.in_proj(x)
        h1 = self.gnn1(x, edge_index, edge_type)
        h1 = self.relu1(h1); h1 = self.drop1(h1)

        h2 = self.gnn2(h1, edge_index, edge_type)
        h2 = self.relu2(h2); h2 = self.drop2(h2)

        pair = torch.cat([h1[a_idx], h1[b_idx], h2[a_idx], h2[b_idx]], dim=1)
        return self.fc(pair)

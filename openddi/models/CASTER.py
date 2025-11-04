# models/CASTER.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CASTER(nn.Module):
    """
    CASTER baseline (张量分解/双线性打分风格)：
      X -> 药物嵌入 e_i
      每种关系 k 学一个向量 r_k
      logits_k(i,j) = < (e_i ⊙ r_k), e_j >
    构造签名与现有 model_manager 一致：
      (feature, hidden1, hidden2, num_relations, num_classes, dropout)
    其中 hidden2 用作最终嵌入维度 d。
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1    = int(hidden1)
        self.dim     = int(hidden2)      # 嵌入维度 d
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # 由节点特征 X 得到药物嵌入 e（两层 MLP，更贴近“特征耦合分解”思想）
        self.enc = nn.Sequential(
            nn.Linear(self.feature, self.hid1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid1, self.dim)
        )

        # 每个关系 k 的对角“变换”向量 r_k（CASTER/DistMult 思想）
        self.rel = nn.Parameter(torch.randn(self.num_classes, self.dim))

        # 可选偏置（药物/关系）
        self.bias_e = nn.Parameter(torch.zeros(self.dim))
        self.bias_r = nn.Parameter(torch.zeros(self.num_classes))

        # 缓存节点特征矩阵 X（与其它 baseline 的 bind_graph 约定一致）
        self.register_buffer("x_cache", None)

    def bind_graph(self, data_graph):
        # 只需要 X，不使用边
        self.x_cache = data_graph.x

    def _node_embed(self, X):
        e = self.enc(X)                   # [N, d]
        # e = F.normalize(e + self.bias_e, dim=-1)  # 稳定训练
        return e

    def forward(self, graph_or_none, idx_batch):
        """
        graph_or_none: 可为 None（已 bind）或包含 x 的 Data
        idx_batch: (i_idx, j_idx, y)
        返回 logits: [B, K]
        """
        X = graph_or_none.x if (graph_or_none is not None) else self.x_cache
        assert X is not None, "CASTER 需要节点特征 X，请先在 Trainer 中 bind_graph(dataset.data_graph)"

        e = self._node_embed(X)           # [N, d]

        device = e.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        e_i = e[i_idx]                    # [B, d]
        e_j = e[j_idx]                    # [B, d]

        # 计算所有关系的 logits：DistMult 风格 (e_i ⊙ r_k) · e_j
        # 等价：先对 e_i 扩展 [B, 1, d]，与 rel [K, d] 做逐元素乘，再与 e_j 点积
        # B, d = e_i.size()
        R = self.rel                      # [K, d]
        # [B, K, d]
        eiR = e_i.unsqueeze(1) * R.unsqueeze(0)
        # [B, K]
        logits = (eiR * e_j.unsqueeze(1)).sum(-1) + self.bias_r.unsqueeze(0)
        # logits = (eiR * e_j.unsqueeze(1)).sum(-1)

        return logits

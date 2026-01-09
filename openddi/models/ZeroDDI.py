# ZeroDDI.py  — 动态原型（带梯度）版：显著提升多类别F1的必要改动
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ZeroDDI(nn.Module):
    """
    ZeroDDI: 将对偶表示 z_ij 与事件语义原型 U_k 映射到同一单位球面，
    logits = (z · U^T) / tau

    增强点：
    - 动态原型：U = normalize(sem_proj(S_raw)) 在 forward 内实时计算（带梯度）
      -> 语义投影 sem_proj 参与分类损失反向传播
    - 特征稳态：LayerNorm + Normalize
    - 较稳的温度初值 tau=0.2（可学习）
    """
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes   = int(num_classes)
        self.hid1          = int(hidden1)
        self.hid2          = int(hidden2)
        self.feature       = int(feature)
        self.dropout       = float(dropout)

        # ---- 节点编码（2层GCN）----
        self.gnn1 = GCNConv(self.feature, self.hid1)
        self.gnn2 = GCNConv(self.hid1, self.hid2)
        self.node_ln = nn.LayerNorm(self.hid2)

        # ---- pair-wise 读出 ----
        pair_in = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.pair_ln_in = nn.LayerNorm(pair_in)
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_in, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.hid2),
        )
        self.pair_ln_out = nn.LayerNorm(self.hid2)

        # ---- 温度（从0.2起步，更稳；可学习）----
        self.tau = nn.Parameter(torch.tensor(0.2), requires_grad=True)

        # ---- 事件语义 ----
        # 原始事件语义 S_raw 会通过 update_event_U(raw) 注入并缓存为 buffer
        self.register_buffer("S_raw", None)    # [K, d_e]
        # 语义投影器：第一次拿到 S_raw 时按 d_e 动态构建
        self.sem_in_dim = None
        self.sem_proj   = None

        # 为了兼容 Trainer 的均匀化正则，这里仍保留 self.U 这个“可读”buffer，
        # 但它在 forward 内会被用最新的 sem_proj(S_raw) 更新为 detach 后的副本。
        self.register_buffer("U", None)        # [K, hid2] (detach)

        # ---- 图缓存（bind_graph时写入，与模型device一致）----
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    # ----------------- 工具 -----------------
    def _dev(self):
        return next(self.parameters()).device

    @staticmethod
    def _pair_features(h, i_idx, j_idx):
        hi = h[i_idx]; hj = h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def _encode_nodes(self, x, edge_index):
        h = self.gnn1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)
        # 稳态：LayerNorm + Dropout
        h = self.node_ln(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N, hid2]

    # ----------------- 供 Trainer 调用 -----------------
    @torch.no_grad()
    def update_event_U(self, event_sem: torch.Tensor):
        """
        仅负责缓存原始事件语义到 self.S_raw，并（若必要）初始化 sem_proj。
        注意：不在这里生成 U（避免 no_grad 使得 sem_proj 无梯度）。
        """
        device = self._dev()
        assert event_sem is not None, "event_sem 为空"
        K, d_e = event_sem.shape
        assert K == self.num_classes, f"事件数 {K} 必须与 num_classes={self.num_classes} 一致"

        self.S_raw = event_sem.to(device).float()  # [K, d_e]

        if (self.sem_proj is None) or (self.sem_in_dim != int(d_e)):
            self.sem_in_dim = int(d_e)
            mid = max(self.hid2, 128)
            self.sem_proj = nn.Sequential(
                nn.Linear(self.sem_in_dim, mid), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(mid, self.hid2),
            ).to(device)

        # 初始化时做一次 U（仅供早期正则/日志；训练时 forward 会覆盖）
        U0 = F.normalize(self.sem_proj(self.S_raw), dim=-1)
        self.U = U0.detach()

    def bind_graph(self, data_graph):
        device = self._dev()
        self.x_cache = data_graph.x.to(device)
        self.edge_index_cache = data_graph.edge_index.to(device, dtype=torch.long)

    # ----------------- 前向（动态原型 + 归一化） -----------------
    def forward(self, graph_or_none, idx_batch):
        """
        返回: (logits [B,K], z [B,hid2])
        """
        device = self._dev()

        if graph_or_none is not None:
            x = graph_or_none.x.to(device)
            edge_index = graph_or_none.edge_index.to(device, dtype=torch.long)
        else:
            assert self.x_cache is not None and self.edge_index_cache is not None, \
                "未绑定图且未传入 graph"
            x = self.x_cache
            edge_index = self.edge_index_cache

        # 1) 节点编码
        h = self._encode_nodes(x, edge_index)                  # [N, hid2]

        # 2) 对偶表征
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_in = self._pair_features(h, i_idx, j_idx)         # [B, 4*hid2]
        pair_in = self.pair_ln_in(pair_in)
        z = self.pair_proj(pair_in)                            # [B, hid2]
        z = self.pair_ln_out(z)
        z = F.normalize(z, dim=-1)

        # 3) 动态原型（带梯度）
        assert self.S_raw is not None, "S_raw 未设置，请先调用 update_event_U()"
        U = self.sem_proj(self.S_raw)                          # [K, hid2]
        U = F.normalize(U, dim=-1)
        # 更新可读 buffer（供 uniformity 正则用）
        self.U = U.detach()

        # 4) 余弦 logits
        logits = torch.matmul(z, U.t()) / self.tau.clamp_min(1e-6)  # [B, K]
        return logits, z

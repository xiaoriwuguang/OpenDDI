# ZeroDDI.py  — Dynamic Prototype (with gradient) version: Necessary modifications to significantly improve multi-class F1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ZeroDDI(nn.Module):
    """
    ZeroDDI: Maps dual representation z_ij and event semantic prototypes U_k to the same unit sphere.
    logits = (z · U^T) / tau
    
    Enhancements:
    
    - Dynamic prototypes: U = normalize(sem_proj(S_raw)) computed in real-time during forward pass (with gradient)
    - Semantic projection sem_proj participates in classification loss backpropagation
    - Feature stability: LayerNorm + Normalize
    - Stable initial temperature value tau=0.2 (learnable)
    
    Args:
        feature (int): Input feature dimension
        hidden1 (int): First hidden layer dimension
        hidden2 (int): Second hidden layer dimension
        num_relations (int): Number of relation types
        num_classes (int): Number of output classes
        dropout (float): Dropout rate, default 0.3
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

        # ---- Node encoding (2-layer GCN) ----
        self.gnn1 = GCNConv(self.feature, self.hid1)
        self.gnn2 = GCNConv(self.hid1, self.hid2)
        self.node_ln = nn.LayerNorm(self.hid2)

        # ---- Pair-wise readout ----
        pair_in = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.pair_ln_in = nn.LayerNorm(pair_in)
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_in, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.hid2),
        )
        self.pair_ln_out = nn.LayerNorm(self.hid2)

        # ---- Temperature (starting from 0.2, more stable; learnable) ----
        self.tau = nn.Parameter(torch.tensor(0.2), requires_grad=True)

        # ---- Event semantics ----
        # Raw event semantics S_raw will be injected and cached as buffer via update_event_U(raw)
        self.register_buffer("S_raw", None)    # [K, d_e]
        # Semantic projector: dynamically built when S_raw is first obtained according to d_e
        self.sem_in_dim = None
        self.sem_proj   = None

        # For compatibility with Trainer's uniformity regularization, retain self.U as a "readable" buffer,
        # but it will be updated in forward with the latest sem_proj(S_raw) as a detached copy.
        self.register_buffer("U", None)        # [K, hid2] (detach)

        # ---- Graph caching (written during bind_graph, consistent with model device) ----
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    # ----------------- Utilities -----------------
    def _dev(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    @staticmethod
    def _pair_features(h, i_idx, j_idx):
        """
        Generate pair features from node representations.
        
        Args:
            h: Node representations
            i_idx: Indices for first drug in pairs
            j_idx: Indices for second drug in pairs
            
        Returns:
            torch.Tensor: Concatenated pair features
        """
        hi = h[i_idx]; hj = h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def _encode_nodes(self, x, edge_index):
        """
        Encode nodes using GCN layers.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            
        Returns:
            torch.Tensor: Encoded node representations
        """
        h = self.gnn1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)
        # Stability: LayerNorm + Dropout
        h = self.node_ln(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N, hid2]

    # ----------------- Called by Trainer -----------------
    @torch.no_grad()
    def update_event_U(self, event_sem: torch.Tensor):
        """
        Only responsible for caching raw event semantics to self.S_raw and (if necessary) initializing sem_proj.
        Note: Does not generate U here (to avoid no_grad preventing sem_proj gradient flow).
        
        Args:
            event_sem (torch.Tensor): Event semantic tensor [K, d_e]
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

        # Initialize U once (only for early regularization/logging; will be overwritten in forward during training)
        U0 = F.normalize(self.sem_proj(self.S_raw), dim=-1)
        self.U = U0.detach()

    def bind_graph(self, data_graph):
        """
        Bind graph data to model for caching.
        
        Args:
            data_graph: Graph data to cache
        """
        device = self._dev()
        self.x_cache = data_graph.x.to(device)
        self.edge_index_cache = data_graph.edge_index.to(device, dtype=torch.long)

    # ----------------- Forward (dynamic prototypes + normalization) -----------------
    def forward(self, graph_or_none, idx_batch):
        """
        Forward pass of the model.
        
        Args:
            graph_or_none: torch_geometric.data.Data or None (uses cached graph if None)
            idx_batch: Batch indices for drug pairs
            
        Returns:
            tuple: (logits [B,K], z [B,hid2])
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

        # 1) Node encoding
        h = self._encode_nodes(x, edge_index)                  # [N, hid2]

        # 2) Dual representation
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_in = self._pair_features(h, i_idx, j_idx)         # [B, 4*hid2]
        pair_in = self.pair_ln_in(pair_in)
        z = self.pair_proj(pair_in)                            # [B, hid2]
        z = self.pair_ln_out(z)
        z = F.normalize(z, dim=-1)

        # 3) Dynamic prototype (with gradient)
        assert self.S_raw is not None, "S_raw 未设置，请先调用 update_event_U()"
        U = self.sem_proj(self.S_raw)                          # [K, hid2]
        U = F.normalize(U, dim=-1)
        # Update readable buffer (for uniformity regularization)
        self.U = U.detach()

        # 4) Cosine logits
        logits = torch.matmul(z, U.t()) / self.tau.clamp_min(1e-6)  # [B, K]
        return logits, z
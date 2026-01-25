# models/MMDGDTI.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # Can be replaced with GATConv/GCNConv

class ModalAttnFusion(nn.Module):
    """
    Multi-modal attention fusion.
    
    For each modality: x_m -> proj_m(x_m) in R^d
    Scoring: s_m = w_m^T x_m
    Attention: α = softmax([s_m]); h = sum_m α_m * ReLU(proj_m(x_m))
    """
    
    def __init__(self, modal_dims, out_dim, dropout=0.0):
        """
        Initialize ModalAttnFusion.
        
        Args:
            modal_dims: List of dimensions for each modality
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.modal_dims = list(map(int, modal_dims))
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)

        self.proj = nn.ModuleList([nn.Linear(d, self.out_dim) for d in self.modal_dims])
        self.scor = nn.ModuleList([nn.Linear(d, 1)           for d in self.modal_dims])

    def forward(self, x, splits):
        """
        Forward pass of ModalAttnFusion.
        
        Args:
            x: Input tensor [N, sum(d_m)]
            splits: Split indices [0, d1, d1+d2, ..., sum]
            
        Returns:
            Tuple of (fused features, attention weights)
        """
        # x: [N, sum(d_m)]; splits: [0, d1, d1+d2, ..., sum]
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
        return F.dropout(h, p=self.dropout, training=self.training), A  # Return attention for reference

class MMDGDTI(nn.Module):
    """
    MMDG-DTI baseline (adapted for DDI).
    
    Architecture:
    - Modal attention fusion -> GraphSAGE (two layers) -> Pair representation -> MLP classification
    - Signature construction consistent with model_manager
    """
    
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize MMDGDTI model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension
            hidden2: Second hidden layer dimension
            num_relations: Number of relation types (unused in current implementation)
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.feature = int(feature)
        self.hid1    = int(hidden1)
        self.hid2    = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # --- Modal splitting (can be injected at runtime via set_modal_splits) ---
        self.modal_dims = [self.feature]  # Default to single modality (after integration)
        self.register_buffer("splits", torch.tensor([0, self.feature], dtype=torch.long))

        # --- Multi-modal fusion (attention) ---
        self.fuse = ModalAttnFusion(self.modal_dims, out_dim=self.hid1, dropout=self.dropout)

        # --- Graph encoding (two-layer SAGE, robust and concise; can be replaced with GAT/GCN/RGCN) ---
        self.gnn1 = SAGEConv(self.hid1, self.hid2)
        self.gnn2 = SAGEConv(self.hid2, self.hid2)

        # --- Pair head (symmetric): [hi, hj, |hi-hj|, hi*hj] -> logits ---
        pair_in = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, mid),     nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.num_classes)
        )

        # Graph caching
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    # Called during Trainer initialization to inform about the dimensions of each modality (consistent with the CSV concatenation order)
    def set_modal_splits(self, split_str: str):
        """
        Set modal splits at runtime (consistent with CSV concatenation order).
        
        Args:
            split_str: Comma-separated string of modal dimensions
        """
        if not split_str:  # Keep single modality
            return
        dims = [int(s) for s in split_str.split(',') if s.strip()]
        assert sum(dims) == self.feature, f"modal_splits 之和需等于 feature={self.feature}, got {sum(dims)}"
        self.modal_dims = dims
        # Rebuild fusion layer
        self.fuse = ModalAttnFusion(self.modal_dims, out_dim=self.hid1, dropout=self.dropout).to(next(self.parameters()).device)
        # Build boundaries
        acc = [0]
        for d in self.modal_dims: acc.append(acc[-1] + d)
        self.splits = torch.tensor(acc, dtype=torch.long, device=next(self.parameters()).device)

    def bind_graph(self, data_graph):
        """
        Bind graph data to the model.
        
        Args:
            data_graph: Graph data object
        """
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, X, edge_index):
        """
        Encode nodes through modal fusion and graph layers.
        
        Args:
            X: Node features
            edge_index: Graph edge indices
            
        Returns:
            Encoded node representations
        """
        # Multi-modal attention fusion
        if self.splits.device != X.device:
            self.splits = self.splits.to(X.device)
        h0, _ = self.fuse(X, self.splits)             # [N, hid1]

        # Graph encoding (two-layer SAGE)
        h = self.gnn1(h0, edge_index); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)                  # [N, hid2]
        return h

    @staticmethod
    def _pair_feat(h, i_idx, j_idx):
        """
        Generate pair features from node embeddings.
        
        Args:
            h: Node embeddings
            i_idx: Indices of first nodes in pairs
            j_idx: Indices of second nodes in pairs
            
        Returns:
            Concatenated pair features
        """
        hi, hj = h[i_idx], h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        """
        Forward pass of the MMDGDTI model.
        
        Args:
            graph_or_none: Can be None (already bound) or Data containing graph information
            idx_batch: (i_idx, j_idx, y) batch indices
            
        Returns:
            logits: Output logits
        """
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
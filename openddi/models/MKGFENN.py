# models/MKGFENN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class FENN(nn.Module):
    """
    Feature-level fusion with gating mechanism.
    
    For each modality m: h_m = ReLU(W_m x_m), g_m = sigmoid(wg_m^T x_m + b_m)
    Fusion: h = Σ_m g_m · h_m
    """
    
    def __init__(self, modal_dims, out_dim, dropout=0.0):
        """
        Initialize FENN module.
        
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
        self.gate = nn.ModuleList([nn.Linear(d, 1)           for d in self.modal_dims])

    def forward(self, x, splits):
        """
        Forward pass of FENN module.
        
        Args:
            x: Input tensor [N, sum(d_m)]
            splits: Split indices [0, d1, d1+d2, ..., sum]
            
        Returns:
            Fused feature representation
        """
        # x: [N, sum(d_m)]; splits: [0, d1, d1+d2, ..., sum]
        parts = []
        for (l, r), pj, gt in zip(zip(splits[:-1], splits[1:]), self.proj, self.gate):
            xm = x[:, l:r]                        # [N, d_m]
            hm = F.relu(pj(xm))                   # [N, out]
            gm = torch.sigmoid(gt(xm))            # [N, 1]
            parts.append(hm * gm)
        h = torch.stack(parts, dim=0).sum(0)      # [N, out]
        return F.dropout(h, p=self.dropout, training=self.training)

class MKGFENN(nn.Module):
    """
    MKG-FENN baseline model.
    
    Architecture:
    - FENN (multi-modal fusion) → RGCN → Pair-MLP → logits
    - Signature construction consistent with model_manager
    """
    
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.5):
        """
        Initialize MKGFENN model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension
            hidden2: Second hidden layer dimension
            num_relations: Number of relation types
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.feature = int(feature)
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # ---- Multi-modal splitting (default single modality, can be set at runtime) ----
        self.modal_dims = [self.feature]
        self.register_buffer("splits", torch.tensor([0, self.feature], dtype=torch.long))
        self.fenn = FENN(self.modal_dims, out_dim=self.hid1, dropout=self.dropout)

        # ---- Graph encoding: two-layer RGCN using edge_type ----
        self.rgcn1 = RGCNConv(self.hid1, self.hid1, num_relations=self.num_relations)
        self.rgcn2 = RGCNConv(self.hid1, self.hid2, num_relations=self.num_relations)

        # ---- Pair → classification ----
        pair_in = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, mid),     nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.num_classes)
        )

        # ---- Graph caching ----
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)
        self.register_buffer("edge_type_cache", None)

    def set_modal_splits(self, split_str: str):
        """
        Set modal splits at runtime (consistent with CSV concatenation order).
        
        Args:
            split_str: Comma-separated string of modal dimensions, e.g., "1024,768,256,128"
        """
        if not split_str:
            return
        dims = [int(s) for s in split_str.split(',') if s.strip()]
        assert sum(dims) == self.feature, f"modal_splits sum must equal feature={self.feature}, got {sum(dims)}"
        self.modal_dims = dims
        # Rebuild FENN and update split boundaries
        self.fenn = FENN(self.modal_dims, out_dim=self.hid1, dropout=self.dropdown if hasattr(self, "dropdown") else self.dropout)
        self.fenn = self.fenn.to(next(self.parameters()).device)
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
        self.edge_type_cache  = getattr(data_graph, "edge_type", None)

    def _encode_nodes(self, x, edge_index, edge_type):
        """
        Encode nodes through FENN fusion and RGCN layers.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            edge_type: Edge relation types
            
        Returns:
            Encoded node representations
        """
        # FENN fusion
        if self.splits.device != x.device:
            self.splits = self.splits.to(x.device)
        h0 = self.fenn(x, self.splits)                 # [N, hid1]
        # RGCN
        h = self.rgcn1(h0, edge_index, edge_type); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.rgcn2(h, edge_index, edge_type)       # [N, hid2]
        return h

    @staticmethod
    def _pair_features(h, i_idx, j_idx):
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
        Forward pass of the MKGFENN model.
        
        Args:
            graph_or_none: Can be None (already bound) or Data containing graph information
            idx_batch: (i_idx, j_idx, y) batch indices
            
        Returns:
            logits: Output logits
        """
        if graph_or_none is not None:
            x = graph_or_none.x; edge_index = graph_or_none.edge_index
            edge_type = getattr(graph_or_none, "edge_type", None)
        else:
            x, edge_index, edge_type = self.x_cache, self.edge_index_cache, self.edge_type_cache
        assert x is not None and edge_index is not None, "Graph not bound or provided"
        assert edge_type is not None, "MKG-FENN requires edge_type (relation types)"

        h = self._encode_nodes(x, edge_index, edge_type)

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pf = self._pair_features(h, i_idx, j_idx)      # [B, 4*hid2]
        logits = self.mlp(pf)                           # [B, K]
        return logits
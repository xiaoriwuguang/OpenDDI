import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TIGER(nn.Module):
    """
    TIGER (Implementation version): Graph encoding + pair representation + MLP classification.
    
    Maintains compatible signature with existing model_manager:
    (feature, hidden1, hidden2, num_relations, num_classes, dropout)
    
    Args:
        feature (int): Input feature dimension
        hidden1 (int): First hidden layer dimension
        hidden2 (int): Second hidden layer dimension
        num_relations (int): Number of relation types (for compatibility)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate, default 0.3
    """
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.feature = int(feature)
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # ---- Graph encoding: Two-layer GAT ----
        heads1, heads2 = 4, 4
        self.gnn1 = GATConv(self.feature, self.hid1 // heads1, heads=heads1, concat=True)
        self.gnn2 = GATConv(self.hid1,   self.hid1 // heads2, heads=heads2, concat=True)

        # ---- Pair representation: concat + abs diff + hadamard ----
        pair_in_dim = 4 * self.hid1
        hid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.num_classes)  # Direct output logits
        )

        # Can bind graph (dataset.data_graph) for Trainer to reduce parameter passing each step
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    def bind_graph(self, data_graph):
        """
        Bind graph data to model for caching.
        
        Args:
            data_graph: Graph data to cache
        """
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, x, edge_index):
        """
        Encode nodes using GAT layers.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            
        Returns:
            torch.Tensor: Encoded node representations
        """
        h = self.gnn1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn2(h, edge_index)
        return h

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
        hi = h[i_idx]
        hj = h[j_idx]
        return torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        """
        Forward pass of the model.
        
        Args:
            graph_or_none: torch_geometric.data.Data or None (uses cached graph if None)
            idx_batch: (i_idx, j_idx, y) - compatible with Base_multi_dataset / Base_multilabel_dataset
            
        Returns:
            torch.Tensor: Output logits [B, K]
        """
        if graph_or_none is not None:
            x, edge_index = graph_or_none.x, graph_or_none.edge_index
        else:
            x, edge_index = self.x_cache, self.edge_index_cache
        assert x is not None and edge_index is not None, "graph 未绑定或传入"

        h = self._encode_nodes(x, edge_index)
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=h.device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=h.device)
        pair_feat = self._pair_features(h, i_idx, j_idx)  # [B, 4*hid2]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.utils.checkpoint import checkpoint

class DDKG(nn.Module):
    """
    DDKG baseline model for drug-drug interaction prediction.
    
    Architecture:
    - Linear dimensionality reduction: feature → hid1
    - Two-layer RGCNConv with num_bases for parameter reduction in multi-relation graphs
    - Checkpointing (without use_reentrant parameter)
    - Pair features → MLP → logits
    """

    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize DDKG model.
        
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

        # Reduce high-dimensional node features to hid1 to decrease edge message dimension
        self.pre = nn.Linear(self.feature, self.hid1)

        # RGCN with low-rank decomposition (bases)
        nb = min(self.num_relations, 16)
        self.rgcn1 = RGCNConv(self.feature, self.hid1,
                              num_relations=self.num_relations,
                              num_bases=nb)
        self.rgcn2 = RGCNConv(self.hid1, self.hid2,
                              num_relations=self.num_relations,
                              num_bases=nb)

        # Pair representation → classification
        pair_in_dim = 4 * self.hid2
        mid = max(self.hid2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, mid), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, mid),         nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(mid, self.num_classes)
        )

        # Graph caching
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)
        self.register_buffer("edge_type_cache", None)

    def bind_graph(self, data_graph):
        """
        Bind graph data to the model for caching.
        
        Args:
            data_graph: Graph data object containing node features, edge indices, and edge types
        """
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index
        self.edge_type_cache  = getattr(data_graph, "edge_type", None)

    def _encode_nodes(self, x, edge_index, edge_type):
        """
        Encode nodes through RGCN layers.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            edge_type: Edge relation types
            
        Returns:
            Encoded node representations
        """
        # Linear dimensionality reduction
        x1 = self.pre(x)
        # Checkpointing (older torch version without use_reentrant parameter)
        h = self.rgcn1(x, edge_index, edge_type)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.rgcn2(h, edge_index, edge_type)
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
        Forward pass of the DDKG model.
        
        Args:
            graph_or_none: Can be None (already bound) or Data containing graph information
            idx_batch: (i_idx, j_idx, y) batch indices
            
        Returns:
            logits: [B, K] output logits
        """
        if graph_or_none is not None:
            x = graph_or_none.x
            edge_index = graph_or_none.edge_index
            edge_type  = getattr(graph_or_none, "edge_type", None)
        else:
            x, edge_index, edge_type = self.x_cache, self.edge_index_cache, self.edge_type_cache

        assert x is not None and edge_index is not None, "Graph not bound or provided"
        assert edge_type is not None, "DDKG requires edge_type (relation types)"

        h = self._encode_nodes(x, edge_index, edge_type)

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_feat = self._pair_features(h, i_idx, j_idx)  # [B, 4*hid2]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits
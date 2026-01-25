import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint

class FeatureMask(nn.Module):
    """
    Feature explainability layer: learns a weight (0-1) for each dimension.
    
    Applies masking in the reduced hid1 space to save GPU memory.
    """

    def __init__(self, in_dim:int):
        """
        Initialize FeatureMask layer.
        
        Args:
            in_dim: Input dimension for the mask
        """
        super().__init__()
        self.mask = nn.Parameter(torch.randn(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature mask to input tensor.
        
        Args:
            x: Input tensor [N, d]
            
        Returns:
            Masked tensor [N, d]
        """
        return x * torch.sigmoid(self.mask)  # [N, d] * [d]

class ExDDI(nn.Module):
    """
    ExDDI baseline model for explainable drug-drug interaction prediction.
    
    Architecture:
    - Linear dimensionality reduction + FeatureMask (hid1 dimension)
    - Single-layer GraphSAGE (hid1→hid2)
    - Checkpointing (without use_reentrant parameter)
    - Pair representation → MLP → logits
    """

    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize ExDDI model.
        
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

        # # Reduce high-dimensional features to hid1 (commented out)
        # self.proj = nn.Linear(self.feature, self.hid1)
        # Apply masking in low-dimensional space
        self.explainer = FeatureMask(self.feature)

        # Single-layer GraphSAGE (more memory efficient)
        self.gnn = SAGEConv(self.feature, self.hid1)

        # Pair → classifier
        pair_in = 4 * self.hid1
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, self.hid2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.hid2),     nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.num_classes)
        )

        # Graph caching
        self.register_buffer("x_cache", None)
        self.register_buffer("edge_index_cache", None)

    def bind_graph(self, data_graph):
        """
        Bind graph data to the model.
        
        Args:
            data_graph: Graph data object containing node features and edge indices
        """
        self.x_cache = data_graph.x
        self.edge_index_cache = data_graph.edge_index

    def _encode_nodes(self, X, edge_index):
        """
        Encode nodes through feature masking and GraphSAGE.
        
        Args:
            X: Node features
            edge_index: Graph edge indices
            
        Returns:
            Encoded node representations
        """
        # z = self.proj(X)                          # [N, hid1] (commented out)
        z = self.explainer(X)                     # [N, hid1]
        # Checkpointing (older torch version without use_reentrant parameter)
        # h = checkpoint(lambda a,b: self.gnn(a,b), z, edge_index)
        h = self.gnn(z,edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h                                   # [N, hid2]

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
        Forward pass of the ExDDI model.
        
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

        h = self._encode_nodes(X, edge_index)

        device = h.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pf = self._pair_feat(h, i_idx, j_idx)
        logits = self.mlp(pf)
        return logits
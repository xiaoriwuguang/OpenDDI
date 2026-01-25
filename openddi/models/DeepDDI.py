# models/DeepDDI.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDDI(nn.Module):
    """
    DeepDDI baseline model for drug-drug interaction prediction.
    
    Architecture:
    
    - No graph convolution, directly uses node feature matrix X
    - Pair representation: [x_i || x_j || absolute difference || element-wise product]
    - MLP -> logits (supports single/multi-label classification)
    """

    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize DeepDDI model.
        
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
        self.hid1 = int(hidden1)
        self.hid2 = int(hidden2)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        pair_in = 4 * self.feature
        hid = max(self.hid1, 256)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.hid2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid2, self.num_classes)  # logits
        )

        # Cache node feature matrix X (from dataset.data_graph.x), interface aligned with TIGER/ZeroDDI
        self.register_buffer("x_cache", None)

    def bind_graph(self, data_graph):
        """
        Bind graph data to the model.
        
        Args:
            data_graph: Graph data object (only uses X, not edge_index)
        """
        # Only uses X, not edge_index
        self.x_cache = data_graph.x

    @staticmethod
    def _pair_features(X, i_idx, j_idx):
        """
        Generate pair features from node features.
        
        Args:
            X: Node feature matrix
            i_idx: Indices of first nodes in pairs
            j_idx: Indices of second nodes in pairs
            
        Returns:
            Concatenated pair features
        """
        xi = X[i_idx]
        xj = X[j_idx]
        return torch.cat([xi, xj, torch.abs(xi - xj), xi * xj], dim=-1)

    def forward(self, graph_or_none, idx_batch):
        """
        Forward pass of the DeepDDI model.
        
        Args:
            graph_or_none: Can be None (already bound) or Data containing node features
            idx_batch: (i_idx, j_idx, y) batch indices
            
        Returns:
            logits: [B, K] output logits
        """
        X = graph_or_none.x if (graph_or_none is not None) else self.x_cache
        assert X is not None, "DeepDDI 需要节点特征表 X，请在 trainer 中先 bind_graph(dataset.data_graph)"
        device = X.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        pair_feat = self._pair_features(X, i_idx, j_idx)  # [B, 4*feature]
        logits = self.mlp(pair_feat)                      # [B, K]
        return logits
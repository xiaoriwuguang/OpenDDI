# models/CASTER.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CASTER(nn.Module):
    """
    CASTER baseline (tensor decomposition/bilinear scoring style).
    
    Architecture:
    - X -> drug embedding e_i
    - Each relation k learns a vector r_k
    - logits_k(i,j) = < (e_i ⊙ r_k), e_j >
    
    Signature construction consistent with existing model_manager:
    - (feature, hidden1, hidden2, num_relations, num_classes, dropout)
    - Where hidden2 is used as the final embedding dimension d.
    """

    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize CASTER model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension
            hidden2: Final embedding dimension (d)
            num_relations: Number of relation types
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.feature = int(feature)
        self.hid1    = int(hidden1)
        self.dim     = int(hidden2)      # Embedding dimension d
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # Get drug embedding e from node features X (two-layer MLP, closer to "feature coupling decomposition" idea)
        self.enc = nn.Sequential(
            nn.Linear(self.feature, self.hid1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid1, self.dim)
        )

        # Diagonal "transformation" vector r_k for each relation k (CASTER/DistMult idea)
        self.rel = nn.Parameter(torch.randn(self.num_classes, self.dim))

        # Optional bias (drug/relation)
        self.bias_e = nn.Parameter(torch.zeros(self.dim))
        self.bias_r = nn.Parameter(torch.zeros(self.num_classes))

        # Cache node feature matrix X (consistent with other baseline's bind_graph convention)
        self.register_buffer("x_cache", None)

    def bind_graph(self, data_graph):
        """
        Bind graph data to the model.
        
        Args:
            data_graph: Graph data containing node features
        """
        # Only need X, not using edges
        self.x_cache = data_graph.x

    def _node_embed(self, X):
        """
        Generate node embeddings from features.
        
        Args:
            X: Input node features
            
        Returns:
            Node embeddings
        """
        e = self.enc(X)                   # [N, d]
        # e = F.normalize(e + self.bias_e, dim=-1)  # Training stabilization
        return e

    def forward(self, graph_or_none, idx_batch):
        """
        Forward pass of the model.
        
        Args:
            graph_or_none: Can be None (already bound) or Data containing x
            idx_batch: (i_idx, j_idx, y) batch indices
            
        Returns:
            logits: [B, K] output logits
        """
        X = graph_or_none.x if (graph_or_none is not None) else self.x_cache
        assert X is not None, "CASTER 需要节点特征 X，请先在 Trainer 中 bind_graph(dataset.data_graph)"

        e = self._node_embed(X)           # [N, d]

        device = e.device
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=device)
        e_i = e[i_idx]                    # [B, d]
        e_j = e[j_idx]                    # [B, d]

        # Calculate logits for all relations: DistMult style (e_i ⊙ r_k) · e_j
        # Equivalent: first expand e_i to [B, 1, d], element-wise multiply with rel [K, d], then dot product with e_j
        # B, d = e_i.size()
        R = self.rel                      # [K, d]
        # [B, K, d]
        eiR = e_i.unsqueeze(1) * R.unsqueeze(0)
        # [B, K]
        logits = (eiR * e_j.unsqueeze(1)).sum(-1) + self.bias_r.unsqueeze(0)
        # logits = (eiR * e_j.unsqueeze(1)).sum(-1)

        return logits
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

__all__ = ["SumGNN"]


class SumGNN(nn.Module):
    """
    SumGNN model with two RGCN layers.
    
    Projects high-dimensional multimodal features to lower dimensions before message passing.
    Pair representation = concat(h1_i, h1_j, h2_i, h2_j) -> fully connected output logits.
    
    Args:
        feature (int): Original multimodal feature dimension (e.g., 2860)
        hidden1 (int): First hidden layer dimension
        hidden2 (int): Second hidden layer dimension  
        num_relations (int): Number of relation types in the graph
        num_classes (int): Number of output classes
        dropout (float): Dropout rate, default 0.3
        proj_dim (int, optional): Projection dimension for input features
        num_bases (int, optional): Number of bases for RGCN weight decomposition
    """
    
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3,
                 proj_dim: int = None, num_bases: int = None):
        super().__init__()
        self.feature = int(feature)          # Original multimodal dimension (2860)
        self.hidden1 = int(hidden1)          # First hidden layer dimension
        self.hidden2 = int(hidden2)          # Second hidden layer dimension
        self.num_rel = max(1, int(num_relations))  # Number of relations
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        # ---- Key: Input dimension reduction (from 2860 to proj_dim, default: hidden1 * 4) ----
        if proj_dim is None:
            proj_dim = max(128, min(512, self.hidden1 * 4))
        self.proj_dim = int(proj_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(self.feature, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ---- RGCN low-rank decomposition: significantly reduces weight memory ----
        if num_bases is None:
            num_bases = min(self.num_rel, 2)  # For single-relation graphs, 2 is sufficient
        self.num_bases = int(num_bases)

        self.gnn1 = RGCNConv(self.feature, self.hidden1,
                             num_relations=self.num_rel, num_bases=self.num_bases)
        self.gnn2 = RGCNConv(self.hidden1, self.hidden2,
                             num_relations=self.num_rel, num_bases=self.num_bases)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

        # Pair representation: [h1_i, h1_j, h2_i, h2_j]
        self.fc = nn.Linear(2 * self.hidden1 + 2 * self.hidden2, self.num_classes)

    def drug_feat(self, emb):
        """Compatibility interface for old code; not used."""
        self.drugfeat = emb

    def forward(self, data_o, idx_batch):
        """
        Forward pass of the model.
        
        Args:
            data_o: Graph data object containing node features, edge indices and types
            idx_batch: Batch indices for drug pairs
            
        Returns:
            torch.Tensor: Output logits for the batch
        """
        x, edge_index, edge_type = data_o.x, data_o.edge_index, data_o.edge_type
        a_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=x.device)
        b_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=x.device)

        # Input dimension reduction to reduce memory usage in edge gathering
        # h0 = self.in_proj(x)
        h1 = self.gnn1(x, edge_index, edge_type)
        h1 = self.relu1(h1); h1 = self.drop1(h1)

        h2 = self.gnn2(h1, edge_index, edge_type)
        h2 = self.relu2(h2); h2 = self.drop2(h2)

        pair = torch.cat([h1[a_idx], h1[b_idx], h2[a_idx], h2[b_idx]], dim=1)
        return self.fc(pair)
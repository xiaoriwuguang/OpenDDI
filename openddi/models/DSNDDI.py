import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv, RGCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class DSNDDI(nn.Module):
    """
    Dual-View Structure-aware Neural Network for Drug-Drug Interaction prediction.

    This model integrates both local drug embeddings and global graph structure
    to predict DDI types.
    """

    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3):
        """
        Initialize the DSNDDI model.

        Args:
            feature: Input feature dimension
            hidden1: Hidden dimension for local and global encoders
            hidden2: Hidden dimension for DDI prediction MLP
            num_relations: Number of relation types in the graph
            num_classes: Number of output classes (DDI types)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.feature = int(feature)

        self.num_edge_features = int(num_classes)
        self.nhid = int(hidden1)
        self.ddi_nhid = int(hidden2)
        self.dropout_ratio = dropout

        # Local View: Process drug embeddings directly
        self.local_mlp = nn.Sequential(
            nn.Linear(self.feature, self.nhid),
            nn.LayerNorm(self.nhid),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio)
        )

        # Global View: Process DDI graph structure
        self.global_conv = RGCNConv(self.feature, self.nhid, num_relations=self.num_relations)
        self.global_norm = nn.LayerNorm(self.nhid)

        # Fusion and Prediction
        # Concatenating Local (nhid) + Global (nhid) for both Head and Tail -> 4 * nhid
        self.mlp = nn.Sequential(
            nn.Linear(self.nhid * 4, self.nhid),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.nhid, self.ddi_nhid),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.ddi_nhid, self.num_classes)  # Output dimension = number of labels
        )

    def forward(self, data_o, idx):
        """
        Forward pass of the model.

        Args:
            data_o: Graph data object containing node features, edge indices and types
            idx: Tuple of (head_indices, tail_indices) for drug pairs

        Returns:
            logits: Predicted logits for each DDI type [batch_size, num_classes]
        """
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        # Local Representation
        x_local = self.local_mlp(x_o)

        # Global Representation
        x_global = self.global_conv(x_o, edge_index, e_type)
        x_global = F.elu(self.global_norm(x_global))

        # Combine Views (Concatenation)
        x_final = torch.cat([x_local, x_global], dim=1)  # [Num_Nodes, nhid * 2]

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)
        
        ent_a = x_final[a_idx]
        ent_b = x_final[b_idx]
        
        # Pair Representation
        pair_vec = torch.cat([ent_a, ent_b], dim=1)  # [Batch, nhid * 4]

        logits = self.mlp(pair_vec)  # [batch_size, num_classes]
        return logits
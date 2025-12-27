# models/MIRACLE.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv

class InteractionPredictor(nn.Module):
    """
    Interaction predictor module for MIRACLE model.
    """
    
    def __init__(self, dg, hidden, k):
        """
        Initialize InteractionPredictor.
        
        Args:
            dg: Input dimension
            hidden: Hidden layer dimension
            k: Output dimension (number of classes)
        """
        super().__init__()
        self.Wl = nn.Linear(dg, hidden)
        self.bl = nn.Parameter(torch.zeros(hidden))
        self.Wp = nn.Linear(hidden, k)
        self.bp = nn.Parameter(torch.zeros(k))

    def forward(self, l):
        """
        Forward pass of InteractionPredictor.
        
        Args:
            l: Input tensor
            
        Returns:
            Predicted interaction scores
        """
        out = F.relu(self.Wl(l) + self.bl)
        out = self.Wp(out) + self.bp
        return out


class MIRACLE(nn.Module):
    """
    MIRACLE model for drug-drug interaction prediction.
    
    This model combines GCN and RGCN layers with interaction prediction.
    """
    
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3, pooling_ratio: float = 0.5):
        """
        Initialize MIRACLE model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension for GCN
            hidden2: Second hidden layer dimension for RGCN
            num_relations: Number of relation types for RGCN
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
            pooling_ratio: Pooling ratio parameter (unused in current implementation)
        """
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature)

        self.num_edge_features = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout

        self.gnn1 = GCNConv(self.feature_dim, self.hidden1)
        self.gnn2 = RGCNConv(self.hidden1, self.hidden2, self.num_relations)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.predictor = InteractionPredictor(self.hidden2, self.hidden2, self.num_classes)

    def forward(self, data_o, idx):
        """
        Forward pass of the MIRACLE model.
        
        Args:
            data_o: Graph data object containing node features, edge indices, and edge types
            idx: Batch indices containing pairs of nodes to process
            
        Returns:
            Predicted interaction scores
        """
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)

        G = x_o 
        x = self.gnn1(G, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        D = self.gnn2(x, edge_index, e_type)

        l_d = D[a_idx] * D[b_idx]
        pred = self.predictor(l_d)

        return pred
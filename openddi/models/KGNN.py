import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import RGCNConv


class KGNN(nn.Module):
    """
    KGNN model for knowledge graph-based drug-drug interaction prediction.
    
    This model uses RGCN layers for graph encoding and linear layer for pair classification.
    """
    
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        """
        Initialize KGNN model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension for RGCN and linear projection
            hidden2: Second hidden layer dimension for RGCN
            num_relations: Number of relation types for RGCN
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.feature = int(feature)
        self.dropout = dropout
        
        self.fc = nn.Linear(self.feature, self.hidden1)
        self.rgcn1 = RGCNConv(self.feature, self.hidden1, num_relations=self.num_relations)
        self.rgcn2 = RGCNConv(self.hidden1, self.hidden2, num_relations=self.num_relations)

        self.linear = nn.Linear(2 * self.hidden2 , self.num_classes)

    def forward(self, data_o, idx):
        """
        Forward pass of the KGNN model.
        
        Args:
            data_o: Graph data object containing node features, edge indices, and edge types
            idx: Batch indices containing pairs of nodes to process
            
        Returns:
            Output logits for the given node pairs
        """
        x_o, edge_index, edge_type = data_o.x, data_o.edge_index, data_o.edge_type
        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)

        # Hierarchical concatenation
        x = self.fc(x_o)

        x1_o = F.relu(self.rgcn1(x_o, edge_index, edge_type))
        xt = F.dropout(x1_o, self.dropout)
        x2_o = self.rgcn2(xt, edge_index, edge_type)

        e_drug_one = x2_o[a_idx]
        e_drug_two = x2_o[b_idx]

        output = self.linear(torch.cat([e_drug_one, e_drug_two], dim = 1))
        return output
import torch
import torch.nn as nn
import torch.optim as optim

class DDIMDL(nn.Module):
    """
    DDIMDL model for multi-modal drug-drug interaction prediction.
    
    This model processes multiple feature modalities independently through separate MLPs
    and combines their outputs through averaging for final prediction.
    """
    
    def __init__(self, features: list, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.5, pooling_ratio: float = 0.3):
        """
        Initialize DDIMDL model.
        
        Args:
            features: List of feature dimensions for each modality
            hidden1: First hidden layer dimension for MLPs
            hidden2: Second hidden layer dimension for MLPs
            num_relations: Number of relation types (unused in current implementation)
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
            pooling_ratio: Pooling ratio parameter (used as dropout in current implementation)
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = [int(f) for f in features] 
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout

        # Create independent MLP for each modality
        self.mlps = nn.ModuleList()
        for f in self.features:
            mlp = nn.Sequential(
                nn.Linear(2 * f, self.hidden1), 
                nn.BatchNorm1d(self.hidden1),
                nn.ReLU(),
                nn.Dropout(self.pooling_ratio),
                nn.Linear(self.hidden1, self.hidden2),
                nn.BatchNorm1d(self.hidden2),
                nn.ReLU(),
                nn.Dropout(self.pooling_ratio),
                nn.Linear(self.hidden2, self.num_classes) 
            )
            self.mlps.append(mlp)

    def forward(self, data_o, idx):
        """
        Forward pass of the DDIMDL model.
        
        Args:
            data_o: Graph data object containing node features, edge indices, and edge types
            idx: Batch indices containing pairs of nodes to process
            
        Returns:
            Averaged output logits from all modalities
        """
        x, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x.device)

        xa = x[a_idx]  # Full feature vector for drug A [batch_size, total_feature_dim]
        xb = x[b_idx]  # Full feature vector for drug B [batch_size, total_feature_dim]

        # Calculate cumulative offsets for splitting modal features
        offsets = [0] + torch.cumsum(torch.tensor(self.features), dim=0).tolist()

        # Collect outputs from each modality
        outputs = []
        for m in range(len(self.features)):
            # Extract features for modality m
            xa_m = xa[:, offsets[m]:offsets[m+1]]
            xb_m = xb[:, offsets[m]:offsets[m+1]]
            x_m = torch.cat((xa_m, xb_m), dim=1)  # Concatenate modality m features of A and B [batch_size, 2 * features[m]]

            # Pass through corresponding MLP
            out_m = self.mlps[m](x_m)
            outputs.append(out_m)

        # Average outputs from all modalities
        final_output = torch.mean(torch.stack(outputs), dim=0)

        return final_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv, RGCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class DSNDDI(nn.Module):
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3):
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
            nn.Linear(self.ddi_nhid, self.num_classes)  # 输出维度=标签数
        )

    def forward(self, data_o, idx):
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        # Local Representation
        x_local = self.local_mlp(x_o)

        # Global Representation
        x_global = self.global_conv(x_o, edge_index, e_type)
        x_global = F.elu(self.global_norm(x_global))

        # Combine Views (Concatenation)
        x_final = torch.cat([x_local, x_global], dim=1) # [Num_Nodes, nhid * 2]

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)
        
        ent_a = x_final[a_idx]
        ent_b = x_final[b_idx]
        
        # Pair Representation
        pair_vec = torch.cat([ent_a, ent_b], dim=1) # [Batch, nhid * 4]

        logits = self.mlp(pair_vec)  # [batch_size, num_classes]
        return logits
    
    

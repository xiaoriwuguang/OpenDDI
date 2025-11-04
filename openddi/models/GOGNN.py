import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv, RGCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GOGNN(nn.Module):
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.5, pooling_ratio: float = 0.5):
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.feature = int(feature)

        self.num_edge_features = int(num_classes)
        self.nhid = int(hidden1)
        self.ddi_nhid = int(hidden2)
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout

        self.conv = RGCNConv(self.feature, 256, num_relations=self.num_relations)


        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, self.num_classes)  # 输出维度=标签数
        )



    def forward(self, data_o, idx):
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)
        
        x = F.relu(self.conv(x_o, edge_index, e_type))
        ent_a = x[a_idx]
        ent_b = x[b_idx]
        pair_vec = torch.cat([ent_a, ent_b], dim=1)

        logits = self.mlp(pair_vec)  # [batch_size, num_classes]
        return logits
    
    

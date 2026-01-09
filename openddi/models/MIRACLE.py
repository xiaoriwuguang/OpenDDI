import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv

class InteractionPredictor(nn.Module):
    def __init__(self, dg, hidden, k):
        super().__init__()
        self.Wl = nn.Linear(dg, hidden)
        self.bl = nn.Parameter(torch.zeros(hidden))
        self.Wp = nn.Linear(hidden, k)
        self.bp = nn.Parameter(torch.zeros(k))

    def forward(self, l):
        out = F.relu(self.Wl(l) + self.bl)
        out = self.Wp(out) + self.bp
        return out


class MIRACLE(nn.Module):
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3, pooling_ratio: float = 0.5):
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
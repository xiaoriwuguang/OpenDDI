# PHGLDDI.py  （修复版：统一 device，兼容旧版 HypergraphConv）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import GCNConv

class GCN_Bottom(nn.Module):
    def __init__(self, hidden=512, feature=300):
        super(GCN_Bottom, self).__init__()
        self.conv1 = GCNConv(feature, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y= self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3]), y[1]

class GIN_Top(torch.nn.Module):
    def __init__(self, fea, hid, hidden=256, train_eps=True):
        super(GIN_Top, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(fea, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hid),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.BatchNorm1d(hid),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hid, hidden)
        self.fc1 = nn.Linear(2 * hidden, 1)
        self.fc2 = nn.Linear(hidden, 1)

    def reset_parameters(self):
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)

        return x

class PHGLDDI(nn.Module):
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int):
        super(PHGLDDI,self).__init__()
        self.BGNN = GCN_Bottom(hidden1, feature)
        self.TGNN = GIN_Top(feature, hidden1, hidden2)
        self.fc = nn.Linear(hidden2, num_classes)

    def forward(self, graph_or_none, idx_batch):
        x, edge_index= graph_or_none.x, graph_or_none.edge_index
        edge_index = edge_index.to(x.device)
        # embs, ed_index = self.BGNN(x, edge_index)
        final = self.TGNN(x, edge_index)
        i_idx = torch.as_tensor(list(idx_batch[0]), dtype=torch.long, device=x.device)
        j_idx = torch.as_tensor(list(idx_batch[1]), dtype=torch.long, device=x.device)
        x1 = final[i_idx]
        x2 = final[j_idx]
        x = torch.mul(x1, x2)
        x = self.fc(x)
        return x

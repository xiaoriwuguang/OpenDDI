import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MUFFIN(nn.Module):

    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3, entity_dim = 128):

        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.feature = int(feature)

        self.num_edge_features = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.entity_dim = int(entity_dim)
        if(self.num_classes > 150) :
            self.entity_dim = 128

        self.activate = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
        
        self.liner = nn.Linear(self.feature, self.entity_dim)
        self.fc1 = nn.Sequential(nn.Linear(29 * 29 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                     nn.ReLU(True))
        if(self.num_classes > 150) :
            self.fc1 = nn.Sequential(nn.Linear(29 * 29 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                     nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), 
                                 nn.ReLU(True))
        # self.layer1 = nn.Sequential(nn.Linear(2 * self.entity_dim, self.num_classes))
        self.layer1 = nn.Sequential(nn.Linear(2 * self.entity_dim, self.hidden1), nn.BatchNorm1d(self.hidden1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.hidden1, self.hidden2), nn.BatchNorm1d(self.hidden2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(self.hidden2, self.num_classes))

    def forward(self, data_o, idx):
        x_o = data_o.x  # (N, feature)
        device = x_o.device

        out1 = self.liner(x_o)                              
        out1 = torch.bmm(out1.unsqueeze(2), out1.unsqueeze(1))
        out1 = out1.unsqueeze(1)                           
        out1 = self.conv1(out1)                             
        out1 = self.conv2(out1)                             
        out1 = out1.view(out1.size(0), -1)                  
        out1 = self.fc1(out1)                               
        out1 = self.fc2(out1)                               

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=device)

        drug1_embed = out1[a_idx] 
        drug2_embed = out1[b_idx]   

        drug_data = torch.cat([drug1_embed, drug2_embed], dim=1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
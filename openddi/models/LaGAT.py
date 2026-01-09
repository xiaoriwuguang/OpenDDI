import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import RGCNConv


class TBA(nn.Module):
    def __init__(self, config):
        super(TBA, self).__init__()
        self.config = config

    def forward(self, inputs):
        drug_embed, neigh_embed = inputs  # drug_embed: (batch, dim), neigh_embed: (batch, n_neighbor^(hop+1), dim)
        n_neighbor = self.config.neighbor_sample_size
        n_shape = int(neigh_embed.shape[1] // n_neighbor)  # 分组数
        # 计算注意力权重
        attention_scores = torch.sum(drug_embed.unsqueeze(1) * neigh_embed, dim=-1, keepdim=True)  # (batch, n, 1)
        weighted_neigh = attention_scores * neigh_embed  # (batch, n, dim)
        # 分组平均
        temp = []
        for i in range(n_shape):
            group = weighted_neigh[:, n_neighbor * i:n_neighbor * (i + 1), :]  # (batch, n_neighbor, dim)
            group_mean = group.mean(dim=1, keepdim=True)  # (batch, 1, dim)
            temp.append(group_mean)
        neighbor_embed = torch.cat(temp, dim=1)  # (batch, n_shape, dim)
        attention_weights = torch.sum(drug_embed.unsqueeze(1) * neigh_embed, dim=-1)  # (batch, n)
        return neighbor_embed, attention_weights

class NeighAggregator(nn.Module):
    def __init__(self, activation='relu', l2_weight=1e-5, name='neigh_aggregator'):
        super(NeighAggregator, self).__init__()
        self.activation = F.relu if activation == 'relu' else torch.tanh
        self.l2_weight = l2_weight
        self.name = name

    def build(self, ent_embed_dim, neighbor_embed_dim):
        # 动态创建线性层
        self.linear = nn.Linear(neighbor_embed_dim, ent_embed_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # 模拟L2正则（实际通过优化器实现）
        self.l2_reg = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, inputs):
        entity, neighbor = inputs  # entity: (batch, 1, dim), neighbor: (batch, n_shape, dim)
        if not hasattr(self, 'linear'):
            self.build(entity.shape[-1], neighbor.shape[-1])
        output = self.linear(neighbor) + self.linear.bias  # (batch, n_shape, ent_dim)
        return self.activation(output)

class GetReceptiveField(nn.Module):
    def __init__(self, config):
        super(GetReceptiveField, self).__init__()
        self.config = config

    def forward(self, x):
        neigh_ent_list = [x]
        neigh_rel_list = []
        batch_size = x.shape[0]

        for i in range(self.config.n_depth):
            indices = neigh_ent_list[-1].long()  # (batch, 1) or (batch, n_neighbor^i)
            new_neigh_ent = self.config.adj_entity[indices].reshape(batch_size, -1)  # (batch, n_neighbor^(i+1))
            new_neigh_rel = self.config.adj_relation[indices].reshape(batch_size, -1)
            neigh_ent_list.append(new_neigh_ent)
            neigh_rel_list.append(new_neigh_rel)

        return neigh_ent_list + neigh_rel_list

class SqueezeLayer(nn.Module):
    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def forward(self, x):
        return x.squeeze(1)


class LaGAT(nn.Module):
    def __init__(self, feature:int, hidden1:int, hidden2:int,
                 num_relations:int, num_classes:int, dropout:float=0.3):
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.feature = int(feature)
        self.dropout = dropout
        # # 嵌入层
        # self.entity_embedding = nn.Embedding(
        #     config.entity_vocab_size, config.ent_embed_dim,
        #     _weight=torch.nn.init.xavier_normal_(torch.empty(config.entity_vocab_size, config.ent_embed_dim))
        # )
        # self.relation_embedding = nn.Embedding(
        #     config.relation_vocab_size, config.ent_embed_dim,
        #     _weight=torch.nn.init.xavier_normal_(torch.empty(config.relation_vocab_size, config.ent_embed_dim))
        # )
        # self.drug_embedding = nn.Embedding(
        #     config.entity_vocab_size, config.ent_embed_dim,
        #     _weight=torch.nn.init.xavier_normal_(torch.empty(config.entity_vocab_size, config.ent_embed_dim))
        # )

        # # 自定义层
        # self.get_receptive_field_one = GetReceptiveField(config, name='receptive_field_drug_one')
        # self.get_receptive_field = GetReceptiveField(config, name='receptive_field_drug')
        self.squeeze_layer = SqueezeLayer()
        self.fc = nn.Linear(self.feature, self.hidden1)
        self.rgcn1 = RGCNConv(self.feature, self.hidden1, num_relations=self.num_relations)
        self.rgcn2 = RGCNConv(self.hidden1, self.hidden2, num_relations=self.num_relations)


        self.linear = nn.Linear(4 * self.hidden1 + 2 * self.hidden2 , self.num_classes)  # 假设86类

    def forward(self, data_o, idx):
        
        x_o, edge_index, edge_type = data_o.x, data_o.edge_index, data_o.edge_type
        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)


        # 层级拼接
        x = self.fc(x_o)

        x1_o = F.relu(self.rgcn1(x_o, edge_index, edge_type))
        xt = F.dropout(x1_o, self.dropout)
        x2_o = self.rgcn2(xt, edge_index, edge_type)

        e_drug_one = torch.cat([x[a_idx], x1_o[a_idx], x2_o[a_idx]], dim=1)
        e_drug_two = torch.cat([x[b_idx], x1_o[b_idx], x2_o[b_idx]], dim=1)

        # # 压缩和softmax
        # drug1_squeeze_embed = self.squeeze_layer(e_drug_one)
        # drug2_squeeze_embed = self.squeeze_layer(e_drug_two)
        
        output = self.linear(torch.cat([e_drug_one, e_drug_two], dim = 1))
        return output

    
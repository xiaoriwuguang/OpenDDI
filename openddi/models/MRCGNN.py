import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)
        torch.nn.init.xavier_uniform_(self.f_k.weight)
        if self.f_k.bias is not None:
            nn.init.zeros_(self.f_k.bias)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None: sc_1 += s_bias1
        if s_bias2 is not None: sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), dim=1)
        return logits

class AvgReadout(nn.Module):
    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, dim=0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, dim=0) / torch.sum(msk)

class MRCGNN(nn.Module):
    """
    num_relations: RGCN 的关系类型数（= 多分类类别数）
    num_classes:   最终预测的类别数（建议与 num_relations 相同）
    """
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.feature = int(feature)

        # 两层 R-GCN
        self.encoder_o1 = RGCNConv(self.feature, self.hidden1, num_relations=self.num_relations)
        self.encoder_o2 = RGCNConv(self.hidden1,  self.hidden2, num_relations=self.num_relations)

        self.read = AvgReadout()
        self.disc = Discriminator(self.hidden2)
        self.dropout = dropout
        self.sigm = nn.Sigmoid()

        # 实体向量：concat(x1_o, x2_o, x_input) => (hidden1 + hidden2 + feature)
        # 配对后再 concat 两端 => 2 * (...)
        pair_in_dim = 2 * (self.hidden1 + self.hidden2 + self.feature)
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, self.num_classes)  # ★ 输出维度=实际类别数
        )

    def forward(self, data_o, data_s, data_a, idx):
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        x_a = data_s.x.to(x_o.device)
        e_type = e_type.long()
        e_type1 = data_a.edge_type.long()

        # 原图表示
        x1_o = F.relu(self.encoder_o1(x_o, edge_index, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, edge_index, e_type)

        # 负样表示
        x1_a = F.relu(self.encoder_o1(x_a, edge_index, e_type))
        x1_a = F.dropout(x1_a, self.dropout, training=self.training)
        x2_a = self.encoder_o2(x1_a, edge_index, e_type)

        # 替代关系编码
        x1_alt = F.relu(self.encoder_o1(x_o, edge_index, e_type1))
        x1_alt = F.dropout(x1_alt, self.dropout, training=self.training)
        x2_alt = self.encoder_o2(x1_alt, edge_index, e_type1)

        # 对比学习读出
        h = self.read(x2_o)
        h = self.sigm(h)
        ret_os   = self.disc(h, x2_o,  x2_a)
        ret_os_a = self.disc(h, x2_o,  x2_alt)

        # batch 边两端索引
        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)

        ent_a = torch.cat([x1_o[a_idx], x2_o[a_idx], x_o[a_idx]], dim=1)
        ent_b = torch.cat([x1_o[b_idx], x2_o[b_idx], x_o[b_idx]], dim=1)
        pair_vec = torch.cat([ent_a, ent_b], dim=1)

        logits = self.mlp(pair_vec)
        return logits, ret_os, ret_os_a, x2_o



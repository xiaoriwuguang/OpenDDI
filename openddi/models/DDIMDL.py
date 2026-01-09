import torch
import torch.nn as nn
import torch.optim as optim

class DDIMDL(nn.Module):
    def __init__(self, features: list, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.5, pooling_ratio: float = 0.3):
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = [int(f) for f in features] 
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout

        # 为每个模态创建一个独立的MLP
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
        x, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x.device)

        xa = x[a_idx]  # 药物 A 的全特征向量 [batch_size, total_feature_dim]
        xb = x[b_idx]  # 药物 B 的全特征向量 [batch_size, total_feature_dim]

        # 计算累积偏移量，用于分割模态特征
        offsets = [0] + torch.cumsum(torch.tensor(self.features), dim=0).tolist()

        # 收集每个模态的输出
        outputs = []
        for m in range(len(self.features)):
            # 提取模态 m 的特征
            xa_m = xa[:, offsets[m]:offsets[m+1]]
            xb_m = xb[:, offsets[m]:offsets[m+1]]
            x_m = torch.cat((xa_m, xb_m), dim=1)  # 拼接 A 和 B 的模态 m 特征 [batch_size, 2 * features[m]]

            # 通过对应的 MLP
            out_m = self.mlps[m](x_m)
            outputs.append(out_m)

        # 对所有模态的输出取平均
        final_output = torch.mean(torch.stack(outputs), dim=0)

        return final_output
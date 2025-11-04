import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
from torch.nn.parameter import Parameter
from torch_geometric.utils import subgraph
from torch_geometric.nn import GCNConv, global_mean_pool

class MVA(nn.Module):
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.feature = int(feature)
        self.dropout_rate = dropout

        # GCN层
        self.gcn1 = GCNConv(self.feature, self.hidden1)
        self.gcn2 = GCNConv(self.hidden1, self.hidden2)

        self.fusionsize = 128
        self.max_d = 50
        self.input_dim_drug = 23532
        self.n_layer = 2
        self.emb_size = 384
        self.dropout_rate = 0
        self.hidden_size = 384
        self.intermediate_size = 1536
        self.num_attention_heads = 4
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1

        self.emb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                               self.num_attention_heads, self.attention_probs_dropout_prob,
                                               self.hidden_dropout_prob)
        self.embed_projection = nn.Linear(self.feature, 128)
        self.fusion = AFF(self.fusionsize)

        # 调整decoder_2以处理GCN输出
        self.decoder_2 = nn.Sequential(
            nn.Linear(self.hidden2, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128)
        )

        self.decoder_trans_mpnn_cat = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
            nn.Linear(64, self.num_classes)
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(50 * 384, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128)
        )

    def forward(self, data_o, idx):
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)

        batch_size = a_idx.size(0)  # 假设为 64
        batch_a = torch.arange(batch_size, device=x_o.device)  # 形状 [64]
        batch_b = torch.arange(batch_size, device=x_o.device)  # 形状 [64]

        # 提取子图
        edge_index_a, _ = subgraph(a_idx, edge_index, relabel_nodes=True, num_nodes=x_o.size(0))
        edge_index_b, _ = subgraph(b_idx, edge_index, relabel_nodes=True, num_nodes=x_o.size(0))

        xa = x_o[a_idx]
        xb = x_o[b_idx]

        # GCN处理子图a
        xa1 = self.gcn1(xa, edge_index_a)
        xa1 = F.relu(xa1)
        xa1 = self.gcn2(xa1, edge_index_a)
        output_1 = global_mean_pool(xa1, batch_a)  # (batch_size, gcn_out_features)
        output_1 = self.decoder_2(output_1)  # (batch_size, 128)

        # GCN处理子图b
        xb1 = self.gcn1(xb, edge_index_b)
        xb1 = F.relu(xb1)
        xb1 = self.gcn2(xb1, edge_index_b)
        output_2 = global_mean_pool(xb1, batch_b)  # (batch_size, gcn_out_features)
        output_2 = self.decoder_2(output_2)  # (batch_size, 128)

        # 处理 (batch_size, embed_dim) 嵌入
        d1_trans_fts_layer1 = self.embed_projection(xa) 
        d2_trans_fts_layer2 = self.embed_projection(xb) 

        # 特征融合
        output1 = self.fusion(d1_trans_fts_layer1, output_1)
        output2 = self.fusion(d2_trans_fts_layer2, output_2)

        final_fts_cat = torch.cat((output1, output2), dim=1)
        result = self.decoder_trans_mpnn_cat(final_fts_cat)

        return result

class AFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(1, inter_channels, kernel_size=(1, 128), stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 128)),
            nn.Conv2d(1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, feature_size = x.size()

        # Reshape x and y as 2D images
        x = x.view(batch_size, 1, 1, feature_size)
        y = y.view(batch_size, 1, 1, feature_size)

        xy = x + y
        xl = self.local_att(xy)
        xg = self.global_att(xy)
        xlg = xl + xg

        wei = self.sigmoid(xlg.squeeze(dim=2).squeeze(dim=2))
        wei_new = wei.squeeze(dim=1)
        wei_new = torch.mean(wei_new, dim=1, keepdim=True)

        # print(wei_new)
        xo = x.squeeze(dim=2).squeeze(dim=2) * wei + y.squeeze(dim=2).squeeze(dim=2) * (1 - wei)
        xo = xo.squeeze(dim=1)
        return xo

# sub-transformer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        input_ids = input_ids.long()
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)#【1.。。50】

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)  # +注意力
        attention_output = self.output(self_output, input_tensor)  # +残差
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # 给向量加了残差和注意力机制
        intermediate_output = self.intermediate(attention_output)  # 给向量拉长
        layer_output = self.output(intermediate_output, attention_output)  # 把向量带着残差压缩回去

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


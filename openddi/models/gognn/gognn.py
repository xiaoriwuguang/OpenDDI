import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_undirected


class GoGNN(torch.nn.Module):
    def __init__(self, args, num_features, nhid, ddi_nhid, pooling_ratio, dropout_ratio, num_rel):
        super(GoGNN, self).__init__()
        self.args = args
        self.num_features = num_features
        # self.ddi_num_features = args.ddi_num_features
        self.nhid = nhid
        self.ddi_nhid = ddi_nhid
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.num_rel = num_rel

        self.conv1 = GCNConv(self.num_features, self.nhid).to(args.device)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv2 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv3 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv_noattn = GCNConv(6 * self.nhid, self.ddi_nhid).to(args.device)
    
        # dropout and edge classifier for supervised training
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * self.ddi_nhid, self.ddi_nhid),
            nn.ReLU(),
            nn.Linear(self.ddi_nhid, self.num_rel)
        )


    def forward(self, data):
        # data: (data_list, ddi_edge_index, ddi_edge_attr or None, [optional placeholders...])
        data_list, ddi_edge_index, *_ = data
        
        # Create Batch object from list of Data objects
        # This is done here to avoid pickling errors in DataLoader workers
        if isinstance(data_list, list):
            batched_data = Batch.from_data_list(data_list)
        else:
            batched_data = data_list

        ddi_edge_index = ddi_edge_index.to(self.args.device)
        
        # Unpack batched data
        x = batched_data.x.to(self.args.device)
        edge_index = batched_data.edge_index.to(self.args.device)
        edge_weight = batched_data.edge_attr.to(self.args.device)
        batch = batched_data.batch.to(self.args.device)

        # Parallel processing of all molecules in the batch
        # Layer 1
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # SAGPooling returns: x, edge_index, edge_attr, batch, perm, score
        x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Layer 2
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Layer 3
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Concatenate features
        modular_feature = torch.cat((x1, x2, x3), dim=1)
        modular_feature = self.dropout(modular_feature)

        # one GCN layer on DDI graph (no edge attributes)
        x = F.relu(self.conv_noattn(modular_feature, ddi_edge_index))

        # edge classification logits for given ddi_edge_index
        source, target = ddi_edge_index
        src_feat = x[source]
        tgt_feat = x[target]
        edge_feat = torch.cat([src_feat, tgt_feat], dim=1)
        logits = self.edge_classifier(edge_feat)

        return logits


    def loss(self, logits, labels):
        """Supervised loss for DDI edge classification.

        If args.matrix in ['multilabel','twosides'] -> BCEWithLogitsLoss (labels float multi-hot)
        else -> CrossEntropyLoss (labels long class indices)
        """
        task = getattr(self.args, 'matrix', 'multiclass')
        if task in ['multilabel', 'twosides']:
            labels = labels.float()
            return nn.BCEWithLogitsLoss()(logits, labels)
        else:
            return nn.CrossEntropyLoss()(logits, labels.long())
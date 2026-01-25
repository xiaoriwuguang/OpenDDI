# This part is adapted from:
# Source: https://github.com/Blair1213/TIGER 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from .GraphTransformer import GraphTransformer
import os

class NodeFeatures(torch.nn.Module):
    """
    Node feature encoder with degree information.
    
    Args:
        degree: Maximum node degree.
        feature_num: Number of input features.
        embedding_dim: Output embedding dimension.
        layer: Number of layers for initialization (default=2).
        type: Type of features ('graph' or 'node') (default='graph').
    """
    def __init__(self, degree, feature_num, embedding_dim, layer=2, type='graph'):
        super(NodeFeatures, self).__init__()

        if type == 'graph': ##代表有feature num
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)  ##将度的值映射成embedding
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        """
        Reset model parameters.
        """
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):
        """
        Encode node features with degree information.
        
        Args:
            data: Graph data object.
            
        Returns:
            torch.Tensor: Encoded node features.
        """
        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())

        return node_feature

class TIGER(torch.nn.Module):
    """
    TIGER model for knowledge graph-enhanced DDI prediction.
    
    Features:
    - Molecular graph transformer
    - Knowledge graph transformer
    - Mutual information maximization
    - Multi-view representation learning
    
    Args:
        max_layer: Maximum number of transformer layers (default=6).
        num_features_drug: Number of drug features (default=78).
        num_nodes: Number of nodes in knowledge graph (default=200).
        num_relations_mol: Number of molecular relation types (default=10).
        num_relations_graph: Number of graph relation types (default=10).
        output_dim: Output dimension (default=64).
        max_degree_graph: Maximum degree in molecular graph (default=100).
        max_degree_node: Maximum degree in knowledge graph (default=100).
        sub_coeff: Subgraph coefficient for loss (default=0.2).
        mi_coeff: Mutual information coefficient for loss (default=0.5).
        dropout: Dropout rate (default=0.2).
        device: Computation device (default='cuda').
        num_rel: Number of relation types for classification.
        args: Configuration arguments.
    """
    def __init__(self, max_layer = 6, num_features_drug = 78, num_nodes = 200, num_relations_mol = 10, num_relations_graph = 10, output_dim=64, max_degree_graph=100, max_degree_node=100, sub_coeff = 0.2, mi_coeff = 0.5, dropout=0.2, device = 'cuda', num_rel = None, args=None):
        super(TIGER, self).__init__()

        print("TIGER Loaded")
        self.device = device
        self.args = args

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, feature_num=num_features_drug, embedding_dim=output_dim, type='graph')
        self.drug_node_feature = NodeFeatures(degree=max_degree_node, feature_num=num_nodes, embedding_dim=output_dim, type='node')

        ##学习的模块
        self.mol_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_mol, dropout= dropout, type='graph')
        self.node_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_graph, dropout=dropout, type='node')
        ##Net用统一的代码就可以了，用type指示是哪种类型的学习，或者分开两个模块，然后两个模块里面集合一些公共的模块

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_rel)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

        self.reserved_loss = 0
        
        self.cdan_dim = output_dim * 2

    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Target device.
            
        Returns:
            TIGER: Model on target device.
        """
        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)

        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)

        self.fc1.to(device)
        self.fc2.to(device)

        self.disc.to(device)
        self.b_xent.to(device)
        return self
    
    def loss(self, pred, label):
        """
        Compute supervised loss for DDI prediction.
        
        Args:
            pred: Model predictions.
            label: Ground truth labels.
            
        Returns:
            torch.Tensor: Combined loss value.
        """
        if self.args.matrix in ['multilabel', 'twosides']:
            return nn.BCEWithLogitsLoss()(pred, label.float()) + self.reserved_loss
        else:
            return nn.CrossEntropyLoss()(pred, label.long()) + self.reserved_loss

    def reset_parameters(self):
        """
        Reset all model parameters.
        """
        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()

        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()

    def forward(self, data):
        """
        Forward pass of TIGER model.
        
        Args:
            data: Tuple containing (drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph)
            
        Returns:
            torch.Tensor: DDI prediction scores.
        """
        drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)

        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        drug1_embedding = self.fc1(torch.cat([drug1_node_embedding, mol1_graph_embedding], dim=-1))
        drug2_embedding = self.fc1(torch.cat([drug2_node_embedding, mol2_graph_embedding], dim=-1))

        final_layer = torch.cat([drug1_embedding, drug2_embedding], dim=-1)
        score = self.fc2(final_layer)

        
        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(self.MI(drug2_embedding, mol2_atom_embedding))
        loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(self.MI(drug2_embedding, drug2_sub_embedding))

        self.reserved_loss = self.mol_coeff* loss_s_m + self.mi_coeff * loss_s_d

        return score

    def MI(self, graph_embeddings, sub_embeddings):
        """
        Compute mutual information between graph and subgraph embeddings.
        
        Args:
            graph_embeddings: Graph-level embeddings.
            sub_embeddings: Subgraph-level embeddings.
            
        Returns:
            torch.Tensor: Discriminator logits.
        """
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        if graph_embeddings.shape[0] > 2:
            idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub)) ##pos
            c_1_list.append(c_1.expand_as(sub)) ##neg
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):
        """
        Compute mutual information loss.
        
        Args:
            logits: Discriminator logits.
            
        Returns:
            torch.Tensor: Binary cross-entropy loss.
        """
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1,-1]), lbl.view([1, -1]))

    def save(self, path):
        """
        Save model state to file.
        
        Args:
            path: Directory path for saving.
            
        Returns:
            str: Path to saved model file.
        """
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path


class Discriminator(nn.Module):
    """
    Discriminator for mutual information estimation.
    
    Args:
        n_h: Hidden dimension size.
    """
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Initialize discriminator weights.
        
        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """
        Forward pass for discriminator.
        
        Args:
            c: Context embeddings.
            h_pl: Positive embeddings.
            h_mi: Negative embeddings.
            s_bias1: Optional bias for positive scores.
            s_bias2: Optional bias for negative scores.
            
        Returns:
            torch.Tensor: Discriminator logits.
        """
        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)
        return logits

def init_params(module, layers=2):
    """
    Initialize module parameters.
    
    Args:
        module: Module to initialize.
        layers: Number of layers for initialization scaling (default=2).
    """
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
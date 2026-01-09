from data.BaseDataset import BaseDataset
import os
import argparse
import torch
import csv
from rdkit import Chem
import networkx as nx
from copy import deepcopy
import random
import math
from torch.utils.data import IterableDataset, DataLoader, Dataset
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

def node_feature_dict(type='onehot'):
    # generate the node feature for each symbol(element)
    # we have 22 different elements in this data set we use the one hot vector
    # or fix_dim 8 dim vector to represent each symbol.
    num_symbols = 22
    symbol_dict = dict()
    keys = ['C', 'Co', 'P', 'K', 'Br', 'B', 'As', 'F', 'Ca', 'La', 'O', 'Au', 'Gd', 'Na', 'Se', 'N', 'Pt', 'S', 'Al',
            'Li', 'Cl', 'I']
    if type == 'onehot':
        for i in range(len(keys)):
            temp = [0] * num_symbols
            temp[i] = 1
            feature = temp
            symbol_dict[keys[i]] = deepcopy(feature)  # just in case
    return symbol_dict

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def node_feature_process(G, feature_type='onehot'):
    """生成节点特征：元素 One-hot + 原子属性 (formal_charge, implicit_valence, degree, ring_atom) + hybridization one-hot。

    最终维度：22 (symbol) + 4 (scalar attrs) + 6 (hybridization one-hot) = 32
    可根据需要在模型侧再进行线性映射。
    """
    feature_dict = node_feature_dict(feature_type)
    symbol_dim = len(next(iter(feature_dict.values()))) if len(feature_dict) else 22

    # 预定义 hybridization 映射
    hybrid_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']
    hybrid_map = {h: i for i, h in enumerate(hybrid_types)}

    symbols = nx.get_node_attributes(G, 'symbol')
    formal_charge = nx.get_node_attributes(G, 'formal_charge')
    implicit_valence = nx.get_node_attributes(G, 'implicit_valence')
    degree_attr = nx.get_node_attributes(G, 'degree')
    ring_atom = nx.get_node_attributes(G, 'ring_atom')
    hybridization = nx.get_node_attributes(G, 'hybridization')

    k = sorted(list(symbols.keys()))
    feats = []
    for key in k:
        sym = symbols.get(key, None)
        base = feature_dict.get(sym, [0] * symbol_dim)
        # 标量特征（简单缩放）
        fc = float(formal_charge.get(key, 0)) / 5.0          # formal_charge 范围通常较小
        iv = float(implicit_valence.get(key, 0)) / 8.0       # 经验缩放
        deg = float(degree_attr.get(key, 0)) / 8.0
        ring = 1.0 if bool(ring_atom.get(key, False)) else 0.0
        # hybridization one-hot
        hyb_raw = str(hybridization.get(key, 'OTHER')).upper()
        hyb_key = 'OTHER'
        for cand in ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']:
            if cand in hyb_raw:
                hyb_key = cand
                break
        hyb_vec = [0.0] * len(hybrid_types)
        hyb_vec[hybrid_map[hyb_key]] = 1.0
        full_vec = base + [fc, iv, deg, ring] + hyb_vec  # 22 +4 +6 =32
        feats.append(full_vec)

    num_nodes = len(k)
    batch = torch.zeros(num_nodes, dtype=torch.int32)
    return torch.tensor(feats, dtype=torch.float32), batch

def edge_preprocess(G):
    edge_weight = []
    edge_1 = []
    edge_2 = []
    bond_types_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1.5}
    bonds = nx.get_edge_attributes(G, 'bond_type')
    edge_index = list(bonds.keys())
    for edge in edge_index:
        edge_weight.append(bond_types_dict[str(bonds[edge])])  # change the name of bond to the string to match the dict
        edge_1.append(edge[0])
        edge_2.append(edge[1])
        edge_1.append(edge[1])
        edge_2.append(edge[0])
        edge_weight.append(bond_types_dict[str(bonds[edge])])
    edge_output = [edge_1, edge_2]
    return torch.LongTensor(edge_output), torch.Tensor(edge_weight)

class EdgeDataset(Dataset):

    def __init__(self, output, edges, labels, dt):
        super().__init__()
        self.output = output
        self.edges = edges
        self.labels = labels
        self.dt = dt

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx: int):
        # 返回当前样本的 (u_id, v_id, label)，在 collate 中统一映射与堆叠
        u_id = self.edges[idx, 0]
        v_id = self.edges[idx, 1]
        label = self.labels[idx]
        return u_id, v_id, label

def build_smile_graph(modular_file_name, feature_type='onehot'):
    id_graph_dict = {}
    with open(modular_file_name, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            id_str = str(row[0]).strip()
            smiles = ''
            if len(row) > 1:
                smiles = row[1].strip()
            if smiles == '':
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            graph = mol_to_nx(mol)
            id_graph_dict[id_str] = graph

    output = dict()
    for ids, graph in id_graph_dict.items():
        node_feature, batch = node_feature_process(graph, feature_type)
        edge_index, edge_weight = edge_preprocess(graph)
        output[ids] = [node_feature, edge_index, edge_weight, batch]
    
    return output

def make_collate_fn(output_dict, task_type='multiclass'):
    """
    Modified collate_fn to return batched graph data for the current batch only.
    This fixes the computational waste and allows parallel processing of molecular graphs.
    """
    # Determine feature dimension from the first available graph, default to 32
    feature_dim = 32
    if len(output_dict) > 0:
        feature_dim = output_dict[next(iter(output_dict))][0].shape[1]
    
    def _collate(samples):
        # samples: list of tuples (u_id, v_id, label)
        
        # 1. Identify unique nodes in this batch
        unique_nodes = set()
        for u_id, v_id, _ in samples:
            unique_nodes.add(u_id)
            unique_nodes.add(v_id)
        
        # Sort to ensure deterministic order
        batch_node_list = sorted(list(unique_nodes))
        id2idx = {k: i for i, k in enumerate(batch_node_list)}
        
        # 2. Construct Data objects for the batch
        data_list = []
        for node_id in batch_node_list:
            if node_id in output_dict:
                # output_dict[node_id] is [x, edge_index, edge_weight, batch]
                x, edge_index, edge_weight, _ = output_dict[node_id]
                # Create PyG Data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
            else:
                # Placeholder for missing graph data
                # Create a single isolated node with zero features
                x = torch.zeros((1, feature_dim), dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty((0,), dtype=torch.float32)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
                
            data_list.append(data)

        # 3. Batch the molecular graphs
        # We return the list of Data objects directly to avoid pickling errors with Batch objects
        # in multiprocessing. The Batch object will be created inside the model's forward pass.
        batched_data = data_list

        # 4. Re-index the DDI edges to point to the new indices in the batch
        us, vs, lbls = [], [], []
        for u_id, v_id, label in samples:
            if (u_id in id2idx) and (v_id in id2idx):
                us.append(id2idx[u_id])
                vs.append(id2idx[v_id])
                lbls.append(label)

        if len(us) == 0:
            edges = torch.empty((2, 0), dtype=torch.long)
            if task_type == 'multiclass':
                labels = torch.empty((0,), dtype=torch.long)
            else:
                labels = torch.empty((0, getattr(samples[0][2], 'shape', [0])[-1] if len(samples) > 0 else 0), dtype=torch.float32)
        else:
            edges = torch.tensor([us, vs], dtype=torch.long)
            if task_type == 'multiclass':
                labels = torch.as_tensor(np.array(lbls), dtype=torch.long)
            else:
                labels = torch.as_tensor(np.array(lbls), dtype=torch.float32)

        return [batched_data, edges, None, labels]

    return _collate

class GoGNN_dataset(BaseDataset):
    def __init__(self,
                 args:argparse.ArgumentParser):
        super().__init__(args)
        self.args = args
        self.num_features = 32
        self.num_edge_features = None

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        super().load_data(val_ratio, test_ratio)

        id_smile = build_smile_graph(
            modular_file_name=os.path.join(self.args.oridata_path, 'id_smiles.csv'),
            feature_type='onehot')
        
        splits = self.build_pairs_labels_splits(val_ratio=val_ratio, test_ratio=test_ratio,
                                                random_seed=getattr(self.args, 'seed', 1),
                                                return_original_ids=True)

        # pairs 是原始字符串 ID，对后续图索引查找保持一致
        train_pairs, train_labels = splits['train']
        val_pairs, val_labels = splits['val']
        test_pairs, test_labels = splits['test']

        all_contained_drugs = set(map(str, np.unique(np.concatenate([train_pairs, val_pairs, test_pairs]).ravel())))

        def pairs_to_np(pairs):
            return np.array([[p[0], p[1]] for p in pairs], dtype=object)

        train_x = pairs_to_np(train_pairs)
        val_x = pairs_to_np(val_pairs)
        test_x = pairs_to_np(test_pairs)

        # 根据数据集类型构造标签张量
        if self.args.matrix in ['multilabel', 'twosides']:
            # 多标签：保持 float32 数组
            train_y = np.array(train_labels, dtype=np.float32)
            val_y = np.array(val_labels, dtype=np.float32)
            test_y = np.array(test_labels, dtype=np.float32)
        else:
            # 多分类：单整数标签
            train_y = np.array(train_labels, dtype=np.int64)
            val_y = np.array(val_labels, dtype=np.int64)
            test_y = np.array(test_labels, dtype=np.int64)

        dt_flag = 'multilabel' if self.args.matrix in ['multilabel', 'twosides'] else 'multiclass'

        train_data = EdgeDataset(output=id_smile, edges=train_x, labels=train_y, dt=dt_flag)
        val_data = EdgeDataset(output=id_smile, edges=val_x, labels=val_y, dt=dt_flag)
        test_data = EdgeDataset(output=id_smile, edges=test_x, labels=test_y, dt=dt_flag)

        params = {
            'batch_size': self.args.batch,
            'shuffle': True,
            'num_workers': int(getattr(self.args, 'workers', 0)),
            'drop_last': False,
            'pin_memory': False,
            'persistent_workers': False,
        }

        if params['num_workers'] > 0:
            params['prefetch_factor'] = 1

        collate_fn = make_collate_fn(id_smile, task_type=dt_flag)
        self.train_loader = DataLoader(train_data, **params, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_data, **params, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_data, **params, collate_fn=collate_fn)

        

        # self.num_edge_features, self.train_loader, self.val_loader, self.test_loader, node_ids, feature_dim = build_dataloaders(
        #     modular_file_name=os.path.join(self.args.oridata_path, 'id_smiles.csv'),
        #     DDI_file_name=self.args.matrix_path,
        #     args=self.args,
        #     device=self.args.device,
        #     val_ratio=val_ratio,
        #     test_ratio=test_ratio,
        #     seed=getattr(self.args, 'seed', 1)
        # )
        # Save explicit node ordering for downstream use / debugging
        self.node_list = all_contained_drugs

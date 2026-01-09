from __future__ import print_function
from data.BaseDataset import BaseDataset
import os
import argparse
import gc
import torch
from tqdm import tqdm
import numpy as np
import json
import copy
from utils import *
from torch_geometric.utils import degree, subgraph
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from rdkit import Chem
import random
import numpy as np


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


class BasicWalker:
    def __init__(self, G, start_nodes, workers):
        self.G = G
        self.workers = workers
        self.start_nodes = start_nodes

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []

        #print('Walk iteration:')
        for walk_iter in range(num_walks):
            #pool = multiprocessing.Pool(processes = )
            #print(str(walk_iter+1), '/', str(num_walks))
            for node in self.start_nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.extend(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))

        return list(set(walks))


class Walker:
    def __init__(self, G, p, q, workers):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[pos][0],
                                               alias_edges[pos][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        look_up_dict = self.look_up_dict
        node_size = self.node_size
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class Node2vec(object):

    def __init__(self, start_nodes, graph, path_length, num_paths, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw: ##deepwalk
            self.walker = BasicWalker(graph, start_nodes, workers=kwargs["workers"])
        else:
            self.walker = Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        self.walks = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)


    def get_walks(self):

        return self.walks

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2]
    mol_rel_index = mol_index[:,2]

    ##在这个位置应该计算的是最短路径
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value
    s_rel[np.where(s_value == 1)] = mol_rel_index  ##将直接相连的关
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    ##c_size:原子的个数
    ##features:每个原子的特征 c_size * 67
    ##edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist(), s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(), max(degrees)

def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)

def smile_to_graph(datapath, ligands):

    smile_graph = {}

    paths = datapath + "/mol_sp.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][6]) if max(smile_graph[s][6]) > max_rel else max_rel
            max_degree = smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree

        return smile_graph, max_rel, max_degree

    smiles_max_node_degree = []
    num_rel_mol_update = 0
    invalid_smiles = []
    single_atom_or_empty = []
    for d, smi in ligands.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # 无法解析的 SMILES：使用占位空图（保持 8 元组结构）
            invalid_smiles.append(d)
            placeholder = (1, [[0 for _ in range(67)]], [[0, 0]], [0], [[0, 0]], [1], [1], 1)
            smile_graph[d] = placeholder
            smiles_max_node_degree.append(1)
            continue
        lg = Chem.MolToSmiles(mol)  # 规范化
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(lg)
        if c_size == 0:  # 单原子 / 无边：也放占位而不是跳过
            single_atom_or_empty.append(d)
            placeholder = (1, [[0 for _ in range(67)]], [[0, 0]], [0], [[0, 0]], [1], [1], 1)
            smile_graph[d] = placeholder
            smiles_max_node_degree.append(1)
            continue
        if len(s_value) > 0 and max(s_value) > num_rel_mol_update:
            num_rel_mol_update = max(s_value)
        smile_graph[d] = (c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg)
        smiles_max_node_degree.append(deg)

    if invalid_smiles:
        print(f"[smile_to_graph] 占位无效 SMILES 数: {len(invalid_smiles)} 示例: {invalid_smiles[:8]}")
    if single_atom_or_empty:
        print(f"[smile_to_graph] 单原子/空图占位数: {len(single_atom_or_empty)} 示例: {single_atom_or_empty[:8]}")

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, num_rel_mol_update, max(smiles_max_node_degree) if smiles_max_node_degree else 0

def read_network(path):

    edge_index = []
    rel_index = []

    flag = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            else:
                flag += 1
                head, rel, tail = line.strip().split("\t")[:3]
                edge_index.append([int(head), int(tail)])
                rel_index.append(int(rel))

        f.close()
    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(len(list(set(rel_index))))

    return num_node, edge_index, rel_index, num_rel

def read_smiles(path):
    """
    Simple reader that returns a dict mapping id->SMILES.
    If `path` is a directory, looks for a file named 'id_smiles.csv' inside it.
    Supports lines like 'id,SMILES' or 'id\tSMILES'. Keeps first occurrence on duplicates.
    """
    # allow passing either a file path or a directory containing id_smiles.csv
    if os.path.isdir(path):
        file_path = os.path.join(path, 'id_smiles.csv')
    else:
        file_path = path

    out = {}
    flag = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw in f:
                if flag == 0:
                    flag = 1
                    continue
                line = raw.strip()
                if not line:
                    continue
                # support both comma and tab; split only on first occurrence
                if ',' in line and '\t' not in line:
                    parts = line.split(',', 1)
                else:
                    parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                id_, seq = parts[0].strip(), parts[1].strip()
                # skip header if present
                if id_.lower() == 'id' or 'smiles' in id_.lower():
                    continue
                if id_ not in out:
                    out[id_] = seq
    except FileNotFoundError:
        print("read_smiles: file not found:", file_path)
    return out

def read_interactions(path, drug_dict):
    interactions = []
    all_drug_in_ddi = []
    positive_drug_inter_dict = {}
    positive_num = 0
    negative_num = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            drug1_id, drug2_id, rel, label = line.strip().split(" ")[:4]
            if drug1_id in drug_dict and drug2_id in drug_dict:
                all_drug_in_ddi.append(drug1_id)
                all_drug_in_ddi.append(drug2_id)
                if float(label) > 0:
                    positive_num += 1
                else:
                    negative_num += 1
                if drug1_id in positive_drug_inter_dict:
                    if drug2_id not in positive_drug_inter_dict[drug1_id]:
                        positive_drug_inter_dict[drug1_id].append(drug2_id)
                        interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
                else:
                    positive_drug_inter_dict[drug1_id] = [drug2_id]
                    interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
        f.close()

    print(positive_num)
    print(negative_num)

    assert negative_num == positive_num

    return np.array(interactions, dtype=int), set(all_drug_in_ddi)

def generate_node_subgraphs(dataset, drug_id, network_edge_index, network_rel_index, num_rel, args):

    edge_index = torch.from_numpy(np.array(network_edge_index).T) ##[2, num_edges]
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row),0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index),1)

    paths = str(dataset) + "/"

    if not os.path.exists(paths):
        os.mkdir(paths)

    subgraphs, max_degree, max_rel_num = rwExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                         sub_num=1, length=32)

    return subgraphs, max_degree, max_rel_num

def rwExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, sub_num, length):

    json_path = shortest_paths + "rw_num_" + str(sub_num) + "_length_" + str(length) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel;

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_index.transpose(1,0).numpy().tolist())
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        # 将 ID 转为整型；若失败则使用占位子图
        try:
            start_node = int(d)
        except Exception:
            # 占位：最小单节点图
            placeholder_sub = ([0], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        # 若起点不在图中，也给占位子图，保证不报错且可训练
        if not my_graph.has_node(start_node):
            placeholder_sub = ([start_node], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        subsets = Node2vec(start_nodes=[start_node], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() ##返回一个list
        # deepwalk 返回的 “walks” 在 BasicWalker 实现里是去重后的若干条 walk 列表，我们需要包含 start_node 的节点集合
        # 这里保持与原有逻辑一致：subsets 作为节点集合使用
        try:
            mapping_id = subsets.index(start_node)
        except ValueError:
            # 极少情况 start_node 不在返回的列表中，仍给出占位
            placeholder_sub = ([start_node], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        mapping_list = [False for _ in range(len((subsets)))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index
        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(int(np.max(s_rel)) if len(s_rel) > 0 else 1)
        node_degree = torch.max(degree(col_sub)).item() if col_sub.numel() > 0 else 1
        max_degree.append(node_degree)

        subgraphs[d] = (subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, node_degree)

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, dt = None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.dt = dt

    def read_drug_info(self, drug_id):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        if edge_index == 0:
            c_size = 1
            features = [[0 for j in range(67)]]
            edge_index = [[0, 0]]
            rel_index = [0]
            sp_edge_index = [[0, 0]]
            sp_value = [1]
            sp_rel = [1]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            #   y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                # y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        # labels = int(self.labels[idx])
        if self.dt == 'multiclass':
            labels = torch.LongTensor([self.labels[idx]])
        else:
            labels = torch.FloatTensor(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id)
        drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subrgraph, labels


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = torch.stack([data[4] for data in data_list]).squeeze(1)

    return batchA, batchB, batchC, batchD, batchE


class TIGER_dataset(BaseDataset):
    def __init__(self,
                 args:argparse.ArgumentParser):
        super().__init__(args)
        self.args = args
        self.interactions = None
        self.labels = None
        self.smile_graph = None
        self.drug_subgraphs = None
        self.data_sta = None

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        super().load_data(val_ratio, test_ratio)

        data_path = self.args.oridata_path

        ligands = read_smiles(data_path)

        # smiles to graphs
        print("load drug smiles graphs!!")
        smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

        print("load networks !!")
        num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "/kgnet.tsv")

        print("load DDI samples!!")
        # 使用 BaseDataset 新增的辅助方法获得配对与标签划分（原始 ID 字符串形式）
        splits = self.build_pairs_labels_splits(val_ratio=val_ratio, test_ratio=test_ratio,
                                                random_seed=getattr(self.args, 'seed', 1),
                                                return_original_ids=True)

        # pairs 是原始字符串 ID，对后续图索引查找保持一致
        train_pairs, train_labels = splits['train']
        val_pairs, val_labels = splits['val']
        test_pairs, test_labels = splits['test']

        # 统计涉及的全部药物 ID，用于生成子图与过滤
        all_contained_drugs = set(map(str, np.unique(np.concatenate([train_pairs, val_pairs, test_pairs]).ravel())))
        # 为 smile_graph 中缺失的药物补充占位空图，避免后续 KeyError（如 'nan' 等无效 ID）
        placeholder_mol = (1, [[0 for _ in range(67)]], [[0, 0]], [0], [[0, 0]], [1], [1], 1)
        missing_smiles = []
        for did in all_contained_drugs:
            if did not in smile_graph:
                smile_graph[did] = placeholder_mol
                missing_smiles.append(did)
        if len(missing_smiles) > 0:
            print(f"[TIGER_dataset] 为 {len(missing_smiles)} 个在 SMILES 映射中缺失的药物填充占位分子图。示例: {missing_smiles[:8]}")


        print("generate subgraphs!!")
        drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(data_path, all_contained_drugs,
                                                                                    network_edge_index, network_rel_index,
                                                                                    num_rel, self.args)

        total_interactions = len(train_pairs) + len(val_pairs) + len(test_pairs)
        data_sta = {
            'num_nodes': num_node + 1,
            'num_rel_mol': num_rel_mol_update + 1,
            'num_rel_graph': num_rel_update + 1,
            'num_interactions': int(total_interactions),
            'num_drugs_DDI': len(all_contained_drugs),
            'max_degree_graph': max_smiles_degree + 1,
            'max_degree_node': int(max_subgraph_degree)+1
        }
        print(data_sta)
        self.data_sta = data_sta
        # 将字符串 ID 对转换为 DataLoader 期望的 numpy 数组
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

        # 构造三份 DTADataset
        # dt 标记用于 __getitem__ 决定标签张量的形状/类型：
        # - 多分类: 使用 'drugbank'（LongTensor 单标签）
        # - 多标签: 使用 'twosides'（FloatTensor 多标签向量）
        dt_flag = 'multilabel' if self.args.matrix in ['multilabel', 'twosides'] else 'multiclass'
        train_data = DTADataset(x=train_x, y=train_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)
        val_data = DTADataset(x=val_x, y=val_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)
        test_data = DTADataset(x=test_x, y=test_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)

        # DataLoader 构建
        self.train_loader = DataLoader(train_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
        self.val_loader = DataLoader(val_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
        self.test_loader = DataLoader(test_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
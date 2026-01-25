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
    """
    Wrapper function for deepwalk walk method.
    
    Args:
        class_instance: Instance of BasicWalker class.
        walk_length: Length of the random walk.
        start_node: Starting node for the walk.
    """
    class_instance.deepwalk_walk(walk_length, start_node)


class BasicWalker:
    """
    Basic random walker for DeepWalk algorithm.
    
    Args:
        G: NetworkX graph.
        start_nodes: List of starting nodes for walks.
        workers: Number of workers (unused in this implementation).
    """
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
    """
    Node2Vec walker with biased random walks.
    
    Args:
        G: Graph object with G, node_size, and look_up_dict attributes.
        p: Return parameter.
        q: In-out parameter.
        workers: Number of workers.
    """
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
    """
    Node2Vec algorithm implementation.
    
    Args:
        start_nodes: List of starting nodes for walks.
        graph: NetworkX graph.
        path_length: Length of each random walk.
        num_paths: Number of walks per node.
        p: Return parameter (default=1.0).
        q: In-out parameter (default=1.0).
        dw: Whether to use DeepWalk instead of Node2Vec (default=False).
        **kwargs: Additional arguments including workers.
    """

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
        """
        Get the generated random walks.
        
        Returns:
            list: List of random walk sequences.
        """
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
    """
    Extract atom features for molecular graph.
    
    Args:
        atom: RDKit atom object.
        
    Returns:
        tuple: (feature_vector, degree) where feature_vector is 78-dimensional.
    """
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
    '''
    Maps inputs not in the allowable set to the last element.
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def single_smile_to_graph(smile):
    """
    Convert SMILES string to molecular graph representation.
    
    Args:
        smile: SMILES string.
        
    Returns:
        tuple: (c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, max_degree)
    """
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

    # The shortest path should be calculated at this location
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value
    s_rel[np.where(s_value == 1)] = mol_rel_index  # Map directly connected relationships to the original edge relationships
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    # c_size: Number of atoms
    # features: The characteristics of each atom c_size * 67
    # edge_index: The edges connecting atoms n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist(), s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(), max(degrees)

def calculate_shortest_path(edge_index):
    """
    Calculate shortest path distances between all node pairs.
    
    Args:
        edge_index: Edge index array of shape (n_edges, 2).
        
    Returns:
        np.array: Array of shape (n_pairs, 3) with [node_i, node_j, distance].
    """
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
    """
    Convert SMILES strings to graph representations and cache to JSON.
    
    Args:
        datapath: Path to save/load cached graph data.
        ligands: Dictionary mapping drug IDs to SMILES strings.
        
    Returns:
        tuple: (smile_graph, max_rel, max_degree) where smile_graph is dictionary of graph data.
    """
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
            # Unparsable SMILES: Using a placeholder empty graph (maintaining the 8-tuple structure)
            invalid_smiles.append(d)
            placeholder = (1, [[0 for _ in range(67)]], [[0, 0]], [0], [[0, 0]], [1], [1], 1)
            smile_graph[d] = placeholder
            smiles_max_node_degree.append(1)
            continue
        lg = Chem.MolToSmiles(mol)  # normalize SMILES
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(lg)
        if c_size == 0:  # Single atom / no edges: also use placeholder instead of skipping
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
    """
    Read knowledge graph network from TSV file.
    
    Args:
        path: Path to TSV file.
        
    Returns:
        tuple: (num_node, edge_index, rel_index, num_rel)
    """
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
    
    Args:
        path: Path to directory or file containing SMILES data.
        
    Returns:
        dict: Dictionary mapping drug IDs to SMILES strings.
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
    """
    Read DDI interactions from file.
    
    Args:
        path: Path to interactions file.
        drug_dict: Dictionary of valid drug IDs.
        
    Returns:
        tuple: (interactions_array, set_of_drugs_in_DDI)
    """
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
    """
    Generate subgraphs for drugs using random walk extraction.
    
    Args:
        dataset: Dataset path.
        drug_id: Set of drug IDs.
        network_edge_index: Knowledge graph edge indices.
        network_rel_index: Knowledge graph relation indices.
        num_rel: Number of relations in KG.
        args: Arguments object.
        
    Returns:
        tuple: (subgraphs_dict, max_degree, max_relation_number)
    """
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
    """
    Extract subgraphs using random walk sampling.
    
    Args:
        drug_id: Set of drug IDs.
        edge_index: Graph edge index tensor.
        rel_index: Relation index tensor.
        shortest_paths: Path for caching.
        num_rel: Number of relations.
        sub_num: Number of walks per node.
        length: Walk length.
        
    Returns:
        tuple: (subgraphs_dict, max_degree, max_relation)
    """
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
        # Convert the ID to an integer; if unsuccessful, use a placeholder image
        try:
            start_node = int(d)
        except Exception:
            # Placeholder: Minimal single-node graph
            placeholder_sub = ([0], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        # If the starting node is not in the graph, also provide a placeholder subgraph to ensure no errors and allow training
        if not my_graph.has_node(start_node):
            placeholder_sub = ([start_node], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        subsets = Node2vec(start_nodes=[start_node], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() # returns list of lists
        # The "walks" returned by DeepWalk, as implemented in BasicWalker, are several lists of walks with duplicate nodes removed. We need a set of nodes that includes the `start_node`
        # Here, we maintain consistency with the original logic: subsets are used as a set of nodes
        try:
            mapping_id = subsets.index(start_node)
        except ValueError:
            # In rare cases, the start_node is not in the returned list, but a placeholder is still provided
            placeholder_sub = ([start_node], [[0, 0]], [0], [True], [[0, 0]], [1], [1], 1)
            subgraphs[d] = placeholder_sub
            num_rel_update.append(1)
            max_degree.append(1)
            continue

        mapping_list = [False for _ in range(len((subsets)))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index
        # Because this involves multi-relation, all edges must be added when adding subgraphs
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
            if sp_value[i] == 1:  # Also ensure all multi-relational edges are in the data
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
    """
    Convert numpy int64 to Python int for JSON serialization.
    
    Args:
        o: Object to convert.
        
    Returns:
        int: Converted integer.
    """
    if isinstance(o, np.int64): return int(o)
    raise TypeError

class DTADataset(InMemoryDataset):
    """
    Dataset class for TIGER model combining molecular and subgraph data.
    
    Args:
        x: Array of drug pairs.
        y: Array of labels.
        sub_graph: Dictionary of drug subgraphs.
        smile_graph: Dictionary of molecular graphs.
        dt: Dataset type flag.
    """
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, dt = None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.dt = dt

    def read_drug_info(self, drug_id):
        """
        Read drug information including molecular and subgraph data.
        
        Args:
            drug_id: Drug ID.
            
        Returns:
            tuple: (data_mol, data_graph) PyTorch Geometric Data objects.
        """
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
        """
        Return the total number of samples in the dataset.
        """
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            tuple: (drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph, labels)
        """
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
    """
    Collate function for batching DTADataset samples.
    
    Args:
        data_list: List of samples.
        
    Returns:
        tuple: Batched PyTorch Geometric Data objects and labels.
    """
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = torch.stack([data[4] for data in data_list]).squeeze(1)

    return batchA, batchB, batchC, batchD, batchE


class TIGER_dataset(BaseDataset):
    """
    TIGER dataset class for knowledge graph-enhanced DDI prediction.
    
    Features:
    - Molecular graph representation from SMILES
    - Knowledge graph subgraph extraction via random walks
    - Dual graph representation (molecular + knowledge graph)
    - Supports both multiclass and multilabel classification
    """
    def __init__(self,
                 args:argparse.ArgumentParser):
        """
        Initialize TIGER dataset.
        
        Args:
            args: Argument parser with configuration parameters.
        """
        super().__init__(args)
        self.args = args
        self.interactions = None
        self.labels = None
        self.smile_graph = None
        self.drug_subgraphs = None
        self.data_sta = None

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Main data loading method.
        
        Args:
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
        """
        super().load_data(val_ratio, test_ratio)

        data_path = self.args.oridata_path

        ligands = read_smiles(data_path)

        # smiles to graphs
        print("load drug smiles graphs!!")
        smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

        print("load networks !!")
        num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "/kgnet.tsv")

        print("load DDI samples!!")
        # Use the new helper methods in BaseDataset to obtain paired data and label assignments (in the form of original ID strings)
        splits = self.build_pairs_labels_splits(val_ratio=val_ratio, test_ratio=test_ratio,
                                                random_seed=getattr(self.args, 'seed', 1),
                                                return_original_ids=True)

        # `pairs` represents the original string IDs, ensuring consistency for subsequent graph index lookups
        train_pairs, train_labels = splits['train']
        val_pairs, val_labels = splits['val']
        test_pairs, test_labels = splits['test']

        # This involves counting all the drug IDs to be used for generating subgraphs and filtering
        all_contained_drugs = set(map(str, np.unique(np.concatenate([train_pairs, val_pairs, test_pairs]).ravel())))
        # Add placeholder empty graphs for missing drugs in `smile_graph` to prevent subsequent `KeyError` (e.g., invalid IDs like 'nan')
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
        # Convert string ID pairs to numpy arrays as expected by DataLoader
        def pairs_to_np(pairs):
            return np.array([[p[0], p[1]] for p in pairs], dtype=object)

        train_x = pairs_to_np(train_pairs)
        val_x = pairs_to_np(val_pairs)
        test_x = pairs_to_np(test_pairs)

        # Construct label tensors based on the dataset type
        if self.args.matrix in ['multilabel', 'twosides']:
            # Multi-label: Keep as float32 array
            train_y = np.array(train_labels, dtype=np.float32)
            val_y = np.array(val_labels, dtype=np.float32)
            test_y = np.array(test_labels, dtype=np.float32)
        else:
            # Multi-class: Single integer label
            train_y = np.array(train_labels, dtype=np.int64)
            val_y = np.array(val_labels, dtype=np.int64)
            test_y = np.array(test_labels, dtype=np.int64)

        # Construct three DTADataset instances
        # dt flag is used in __getitem__ to determine the shape/type of the label tensor:
        # - Multi-class: Use 'drugbank' (LongTensor single label)
        # - Multi-label: Use 'twosides' (FloatTensor multi-label vector)
        dt_flag = 'multilabel' if self.args.matrix in ['multilabel', 'twosides'] else 'multiclass'
        train_data = DTADataset(x=train_x, y=train_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)
        val_data = DTADataset(x=val_x, y=val_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)
        test_data = DTADataset(x=test_x, y=test_y, sub_graph=drug_subgraphs, smile_graph=smile_graph, dt=dt_flag)

        # DataLoader construction
        self.train_loader = DataLoader(train_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
        self.val_loader = DataLoader(val_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
        self.test_loader = DataLoader(test_data, batch_size=self.args.batch, shuffle=True, collate_fn=collate)
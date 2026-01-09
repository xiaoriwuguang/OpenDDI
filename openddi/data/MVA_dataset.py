import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data
from subword_nmt.apply_bpe import BPE
import codecs
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from data.BaseDataset import BaseDataset
import argparse
import os
from utils.config import CODE_DIR

ESPF_path = os.path.join(CODE_DIR,'..','datasets','ESPF')

def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths"""
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

  return intervals

possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']

reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]

intervals = get_intervals(reference_lists)

def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found"""
  try:
    return l.index(e)
  except:
    return len(l)

def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  return features

def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals"""
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]

  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id

def atom_to_id(atom):
  """Return a unique id corresponding to the atom type"""
  features = get_feature_list(atom)
  return features_to_id(features, intervals)

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:

    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)

def drug2emb_encoder(x, dbpe, words2idx_d):
    # Sequence encoder parameter
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([1])
    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

class MolGraphDataset(Dataset):
    def __init__(self, data_df, id2smiles, prediction=False):
        # word
        vocab_path = os.path.join(ESPF_path,'drug_codes_chembl.txt')
        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        sub_csv = pd.read_csv(os.path.join(ESPF_path,'subword_units_map_chembl.csv'))
        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        self.dbpe = dbpe
        self.words2idx_d = words2idx_d

        self.smiles1 = []
        self.smiles2 = []
        self.targets = []

        for _, row in data_df.iterrows():
            if pd.isna(row['id1']) or pd.isna(row['id2']):
                continue
            id1 = int(row['id1'])
            id2 = int(row['id2'])
            
            if id1 in id2smiles and id2 in id2smiles:
                s1 = id2smiles[id1]
                s2 = id2smiles[id2]
                
                if pd.isna(s1) or pd.isna(s2) or s1 == '' or s2 == '':
                    continue
                
                self.smiles1.append(s1)
                self.smiles2.append(s2)
                
                if prediction:
                    # 多标签：保持 float32 数组
                    self.targets.append(float(row['ddi']))
                else:
                    # 多分类：单整数标签
                    self.targets.append(int(row['ddi']))

        self.smiles1 = np.array(self.smiles1)
        self.smiles2 = np.array(self.smiles2)
        if prediction:
             self.targets = np.array(self.targets, dtype=np.float32)
        else:
             self.targets = np.array(self.targets, dtype=np.int64)

    def __getitem__(self, index):

        # num_size = Chem.MolFromSmiles(self.smiles1[index]).GetNumAtoms()
        fts1, adjs1 = smile_to_graph(self.smiles1[index])
        fts2, adjs2 = smile_to_graph(self.smiles2[index])

        num_size = Chem.MolFromSmiles(self.smiles1[index]).GetNumAtoms()
        d1, mask_1 = drug2emb_encoder(self.smiles1[index], self.dbpe, self.words2idx_d)
        d2, mask_2 = drug2emb_encoder(self.smiles2[index], self.dbpe, self.words2idx_d)

        targets = self.targets[index]

        return (fts1, adjs1), (fts2, adjs2), num_size, targets, d1, d2, mask_1, mask_2

    def __len__(self):
        return len(self.smiles1)
    
def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    return node_features, adjacency

def molgraph_collate_fn(data):
    n_samples = len(data)
    (fts1, adjs1), (fts2, adjs2), num_size, targets_0, d1, d2, mask_1, mask_2 = data[0]

    n_nodes_largest_graph_1 = max(map(lambda sample: sample[0][0].shape[0], data))
    n_nodes_largest_graph_2 = max(map(lambda sample: sample[1][0].shape[0], data))

    n_node_fts_1 = fts1.shape[1]
    n_node_fts_2 = fts2.shape[1]

    n_emb = d1.shape[0]
    n_mask = mask_1.shape[0]

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)

    adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
    node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_fts_2)

    num_size_tensor = torch.zeros(n_samples, num_size)
    
    # Determine target shape based on type
    if isinstance(targets_0, (int, np.integer)):
        target_tensor = torch.zeros(n_samples, dtype=torch.long)
    else:
        target_tensor = torch.zeros(n_samples, 1, dtype=torch.float32)

    d1_emb_tensor = torch.zeros(n_samples, n_emb)
    d2_emb_tensor = torch.zeros(n_samples, n_emb)
    mask_1_tensor = torch.zeros(n_samples, n_mask)
    mask_2_tensor = torch.zeros(n_samples, n_mask)


    for i in range(n_samples):
        (fts1, adjs1), (fts2, adjs2), num_size, target, d1, d2, mask_1, mask_2 = data[i]

        n_nodes_1 = adjs1.shape[0]
        n_nodes_2 = adjs2.shape[0]

        num_size_tensor[i] = torch.tensor(num_size)
        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adjs1)
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(fts1)
        adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(adjs2)
        node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(fts2)

        target_tensor[i] = torch.tensor(target)
        d1_emb_tensor[i] = torch.IntTensor(d1)
        d2_emb_tensor[i] = torch.IntTensor(d2)
        mask_1_tensor[i] = torch.tensor(mask_1)
        mask_2_tensor[i] = torch.tensor(mask_2)

    return node_tensor_1, adjacency_tensor_1, node_tensor_2, adjacency_tensor_2, num_size_tensor, target_tensor, d1_emb_tensor, d2_emb_tensor, mask_1_tensor, mask_2_tensor

class MVA_dataset(BaseDataset):
    def __init__(self,
                 args:argparse.ArgumentParser):
        super().__init__(args)
        self.args = args

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        super().load_data(val_ratio, test_ratio)
        smiles_path = self.args.oriSmiles_path
        ddi_path = self.args.matrix_path

        # Load SMILES mapping
        id_smiles_df = pd.read_csv(smiles_path)
        id2smiles = dict(zip(id_smiles_df['id1'], id_smiles_df['SMILES']))
        
        # Load DDI data
        ddi_df = pd.read_csv(ddi_path)
        
        # Shuffle and split
        ddi_df = ddi_df.sample(frac=1, random_state=1).reset_index(drop=True)
        n_total = len(ddi_df)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test
        
        train_df = ddi_df.iloc[:n_train]
        val_df = ddi_df.iloc[n_train:n_train+n_val]
        test_df = ddi_df.iloc[n_train+n_val:]

        # 根据数据集类型构造标签张量
        if self.args.matrix in ['multilabel', 'twosides']:
            # 多标签：保持 float32 数组
            train_dataset = MolGraphDataset(train_df, id2smiles, prediction=True) # prediction=True for float labels
            validation_dataset = MolGraphDataset(val_df, id2smiles, prediction=True)
            test_dataset = MolGraphDataset(test_df, id2smiles, prediction=True)
        else:
            # 多分类：单整数标签
            train_dataset = MolGraphDataset(train_df, id2smiles, prediction=False) # prediction=False for int labels
            validation_dataset = MolGraphDataset(val_df, id2smiles, prediction=False)
            test_dataset = MolGraphDataset(test_df, id2smiles, prediction=False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch,
                                  shuffle=True, collate_fn=molgraph_collate_fn)

        self.val_loader = DataLoader(validation_dataset, batch_size=self.args.batch,
                                       shuffle=False, collate_fn=molgraph_collate_fn)

        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch,
                                 shuffle=True, collate_fn=molgraph_collate_fn)

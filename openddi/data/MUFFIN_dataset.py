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

from torch.utils.data import Dataset, DataLoader

import random
import numpy as np

class MuffinDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

def make_collate_fn(entity_embed, structure_embed, task_type='multiclass', device='cpu'):
    # Move embeddings to shared memory or device once if possible, 
    # but be careful with multiprocessing.
    # Here we keep them as tensors.
    
    def _collate(samples):
        # samples: list of (u, v, label)
        us, vs, lbls = zip(*samples)
        
        # Construct edge_index: [2, batch_size]
        # These are indices into the embedding matrices
        edge_index = torch.tensor([us, vs], dtype=torch.long)
        
        if task_type == 'multiclass':
            labels = torch.tensor(lbls, dtype=torch.long)
        else:
            labels = torch.tensor(lbls, dtype=torch.float32)
            
        # Return format matches MUFFIN.forward expectations:
        # batch_data = [entity_embed, structure_embed, edge_index, labels]
        # Note: We pass None for embeddings to avoid pickling large matrices in multiprocessing.
        # The Trainer is responsible for injecting the embeddings (on device) into the batch.
        return [None, None, edge_index, labels]
    return _collate

class MUFFIN_dataset(BaseDataset):
    def __init__(self,
                 args:argparse.ArgumentParser):
        super().__init__(args)
        self.args = args
        self.entity_dim = 0
        self.structure_dim = 0
        self.entity_pre_embed = None
        self.structure_pre_embed = None

    def load_data(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        super().load_data()
        # 1. Load Pretrained Embeddings
        # Use BaseDataset's helper to load .pt files (id -> embedding)
        entity_path = self.args.embedding_map['drkg']
        structure_path = self.args.embedding_map['smiles']
        
        print(f"Loading entity embeddings from {entity_path}")
        # read_id_embedding_pt returns (id2vec_dict, dim)
        entity_id2vec, self.entity_dim = self.data_loader.read_id_embedding_pt(entity_path)
        
        print(f"Loading structure embeddings from {structure_path}")
        structure_id2vec, self.structure_dim = self.data_loader.read_id_embedding_pt(structure_path)
        
        # 2. Get Data Splits (using BaseDataset's method)
        # This returns original string IDs
        splits = self.build_pairs_labels_splits(val_ratio=val_ratio, test_ratio=test_ratio,
                                                return_original_ids=True)
        
        # 3. Build ID to Index Mapping & Embedding Matrices
        # We need to align entity and structure embeddings to the same index space.
        # We collect ALL unique drug IDs from the dataset splits.
        
        all_pairs = np.concatenate([splits['train'][0], splits['val'][0], splits['test'][0]])
        unique_ids = sorted(list(set(all_pairs.flatten())))
        
        id2idx = {id_str: i for i, id_str in enumerate(unique_ids)}
        num_drugs = len(unique_ids)
        
        # Initialize embedding matrices
        self.entity_pre_embed = torch.zeros((num_drugs, self.entity_dim), dtype=torch.float32)
        self.structure_pre_embed = torch.zeros((num_drugs, self.structure_dim), dtype=torch.float32)
        
        # Fill matrices
        missing_entity = 0
        missing_structure = 0
        
        for id_str, idx in id2idx.items():
            # Entity embedding
            if id_str in entity_id2vec:
                self.entity_pre_embed[idx] = torch.from_numpy(entity_id2vec[id_str])
            else:
                missing_entity += 1
                self.entity_pre_embed[idx] = torch.zeros(self.entity_dim, dtype=torch.float32)
                
            # Structure embedding
            if id_str in structure_id2vec:
                self.structure_pre_embed[idx] = torch.from_numpy(structure_id2vec[id_str])
            else:
                missing_structure += 1
                self.structure_pre_embed[idx] = torch.zeros(self.structure_dim, dtype=torch.float32)
        

        # 4. Convert ID pairs to Index pairs
        def process_split(split_name):
            pairs, labels = splits[split_name]
            idx_pairs = []
            valid_labels = []
            for (u, v), l in zip(pairs, labels):
                if u in id2idx and v in id2idx:
                    idx_pairs.append([id2idx[u], id2idx[v]])
                    valid_labels.append(l)
            return np.array(idx_pairs), np.array(valid_labels)

        train_pairs, train_labels = process_split('train')
        val_pairs, val_labels = process_split('val')
        test_pairs, test_labels = process_split('test')
        
        # 5. Build DataLoaders
        dt_flag = 'multilabel' if self.args.matrix in ['multilabel', 'twosides'] else 'multiclass'
        
        collate_fn = make_collate_fn(self.entity_pre_embed, self.structure_pre_embed, 
                                     task_type=dt_flag, device=self.args.device)
        
        params = {
            'batch_size': self.args.batch,
            'shuffle': True,
            'num_workers': int(getattr(self.args, 'workers', 0)),
            'collate_fn': collate_fn,
            'drop_last': False
        }
        
        # For multiprocessing, we shouldn't pass CUDA tensors if num_workers > 0
        # But here we handle device movement inside collate_fn or after loading.
        if params['num_workers'] > 0:
             params['prefetch_factor'] = 1

        self.train_loader = DataLoader(MuffinDataset(train_pairs, train_labels), **params)
        self.val_loader = DataLoader(MuffinDataset(val_pairs, val_labels), **params)
        self.test_loader = DataLoader(MuffinDataset(test_pairs, test_labels), **params)
import os
import re
import pandas as pd
import numpy as np
import torch
import random
import argparse
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from data.BaseDataset import BaseDataset

class MRCGNN_dataset(BaseDataset):
    """
    Dataset class for MRCGNN model with support for multi-class and multi-label data.
    
    This class extends BaseDataset to provide MRCGNN-specific data loading logic,
    including construction of adversarial samples for contrastive learning.
    
    Args:
        args (argparse.ArgumentParser): Command line arguments containing 
                                       configuration parameters.
    """
    
    def __init__(self,
                 args:argparse.ArgumentParser):
        """
        Initialize the MRCGNN dataset.
        
        Args:
            args (argparse.ArgumentParser): Command line arguments containing 
                                           configuration parameters.
        """
        super().__init__(args)
        self.data_s = None
        self.data_a = None
        self.predict_loader = None
        self.index_to_drug_id = None

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """
        Load data with validation and test splits.
        
        Args:
            val_ratio (float): Ratio of data to use for validation. Defaults to 0.1.
            test_ratio (float): Ratio of data to use for testing. Defaults to 0.2.
        """
        super().load_data(val_ratio, test_ratio)

        if self.args.matrix in ['multilabel', 'twosides']:
            self._load_multilabel_data_additional()
        else :
            self._load_multi_data_additional()

    def _load_multi_data_additional(self):
        """
        Load multi-class data with MRCGNN-specific logic including adversarial sample construction.
        
        This method constructs adversarial samples for contrastive learning by 
        randomly perturbing edges/node features and creates MRCGNN-specific data objects.
        """
        # Get training data from BaseDataset for graph construction
        # Use loaded self.data_o to build additional contrastive learning data

        # Adversarial nodes, used for contrastive learning in MRCGNN, 
        # randomly perturb edges/node features
        features_o = self.data_o.x.detach().cpu().numpy()
        id_perm = np.random.permutation(features_o.shape[0])
        x_a = torch.tensor(features_o[id_perm], dtype=torch.float)

        # Get drug list
        num_drugs = features_o.shape[0]
        y_a = torch.cat((torch.ones(num_drugs, 1), torch.zeros(num_drugs, 1)), dim=1)

        # Get edge and edge type information for contrastive learning
        edge_index_o = self.data_o.edge_index
        edge_types = self.data_o.edge_type

        # Build edge type pair data
        edge_types_pair = []
        for i in range(0, len(edge_types), 2):  # Take one from each bidirectional edge
            r = edge_types[i].item()
            edge_types_pair.extend([r, r])
        edge_types_pair = torch.tensor(edge_types_pair, dtype=torch.int64)

        # Create MRCGNN-specific data objects
        self.data_s = Data(x=x_a, edge_index=edge_index_o, edge_type=edge_types)
        self.data_a = Data(x=self.data_o.x, y=y_a, edge_type=edge_types_pair)


    def _load_multilabel_data_additional(self):
        """
        Load multi-label data with MRCGNN-specific logic including adversarial sample construction.
        
        This method constructs adversarial samples for contrastive learning specifically
        for multi-label classification tasks.
        """
        # Get training data from BaseDataset for graph construction
        # Use loaded self.data_o to build additional contrastive learning data

        # Adversarial nodes, used for contrastive learning in MRCGNN, 
        # randomly perturb edges/node features
        features_o = self.data_o.x.detach().cpu().numpy()
        id_perm = np.random.permutation(features_o.shape[0])
        x_a = torch.tensor(features_o[id_perm], dtype=torch.float)

        # Get drug list
        num_drugs = features_o.shape[0]
        y_a = torch.cat((torch.ones(num_drugs, 1), torch.zeros(num_drugs, 1)), dim=1)

        # Get edge and edge type information for contrastive learning
        edge_index_o = self.data_o.edge_index
        edge_types = self.data_o.edge_type

        # For multi-label, edge_types are all 0
        edge_types_pair = []
        for i in range(0, len(edge_types), 2):  # Take one from each bidirectional edge
            edge_types_pair.extend([0, 0])
        edge_types_pair = torch.tensor(edge_types_pair, dtype=torch.int64)

        # Create MRCGNN-specific data objects
        self.data_s = Data(x=x_a, edge_index=edge_index_o, edge_type=edge_types)
        self.data_a = Data(x=self.data_o.x, y=y_a, edge_type=edge_types_pair)

# Responsible for determining which specific dataset is needed
import argparse
from data.MRCGNN_dataset import MRCGNN_dataset
from data.ZeroDDI_dataset import ZeroDDI_dataset
from data.Unified_dataset import Unified_dataset
from data.TIGER_dataset import TIGER_dataset
from data.GoGNN_dataset import GoGNN_dataset
from data.MUFFIN_dataset import MUFFIN_dataset
from data.MVA_dataset import MVA_dataset
class dataset_manager:
    """
    A manager class for handling different dataset types based on the model.
    
    This class maps model names to their corresponding dataset classes and
    provides functionality to load the appropriate dataset based on the 
    specified model.
    
    Args:
        args (argparse.ArgumentParser): Command line arguments containing 
                                       model specification and other parameters.
    """
    
    def __init__(self,
                 args: argparse.ArgumentParser):
        """
        Initialize the dataset manager.
        
        Args:
            args (argparse.ArgumentParser): Command line arguments containing 
                                           model specification and other parameters.
        """
        self.dataset = None
        self.args = args
        self.dataset_mapping = {"MRCGNN": MRCGNN_dataset,
                                "GOGNN": Unified_dataset,
                                "ZeroDDI": ZeroDDI_dataset,
                                "DDIMDL": Unified_dataset,
                                "TIGER": Unified_dataset, 
                                "ConvLSTM": Unified_dataset,    
                                "MVA": Unified_dataset,
                                "MUFFIN": Unified_dataset,
                                "DeepDDI": Unified_dataset,
                                "DDKG": Unified_dataset,
                                "SumGNN": Unified_dataset,
                                "KGNN": Unified_dataset,
                                "LaGAT": Unified_dataset,
                                "PHGLDDI": Unified_dataset,
                                "MMDGDTI": Unified_dataset,
                                "DSNDDI": Unified_dataset,
                                "ExDDI": Unified_dataset,
                                "MIRACLE": Unified_dataset,
                                "CASTER": Unified_dataset,
                                "MKGFENN": Unified_dataset,
                                }

    def load_dataset(self):
        """
        Load the dataset corresponding to the specified model.
        
        Returns:
            object: An instance of the dataset class corresponding to the model.
        """
        self.dataset = self.dataset_mapping[self.args.model](self.args)
        return self.dataset
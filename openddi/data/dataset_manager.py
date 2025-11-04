#负责判断具体需要哪个dataset
import argparse
from data.MRCGNN_dataset import MRCGNN_dataset
from data.ZeroDDI_dataset import ZeroDDI_dataset
from data.Unified_dataset import Unified_dataset
class dataset_manager:
    def __init__(self,
                 args: argparse.ArgumentParser):
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
                                "ExDDI": Unified_dataset,
                                "MIRACLE": Unified_dataset,
                                "CASTER": Unified_dataset,
                                "MKGFENN": Unified_dataset,
                                }

    def load_dataset(self):
        self.dataset = self.dataset_mapping[self.args.model](self.args)
        return self.dataset


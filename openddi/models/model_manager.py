import argparse
import torch
from models.MRCGNN import MRCGNN
from models.GOGNN import GOGNN
from models.ZeroDDI import ZeroDDI
from models.DDIMDL import DDIMDL    
from models.TIGER import TIGER
from models.ConvLSTM import ConvLSTM
from models.MVA import MVA
from models.MUFFIN import MUFFIN
from models.DeepDDI import DeepDDI
from models.DDKG import DDKG
from models.SumGNN import SumGNN
from models.KGNN import KGNN
from models.LaGAT import LaGAT
from models.PHGLDDI import PHGLDDI
from models.MMDGDTI import MMDGDTI 
from models.ExDDI import ExDDI
from models.MIRACLE import MIRACLE
from models.CASTER import CASTER 
from models.MKGFENN import MKGFENN
from inspect import signature

class model_manager:
    """
    Model manager for handling different model types based on configuration.
    
    This class maps model names to their corresponding model classes and
    provides functionality to load the appropriate model with proper parameters.
    """
    
    def __init__(self,
                 args:argparse):
        """
        Initialize the model manager.
        
        Args:
            args: Command line arguments containing model specification and parameters
        """
        self.args = args    
        self.model_mapping = {"MRCGNN": MRCGNN,
                              "GOGNN" : GOGNN,
                              "ZeroDDI": ZeroDDI,
                              "DDIMDL": DDIMDL,
                              "TIGER": TIGER,
                              "ConvLSTM": ConvLSTM,
                              "MVA": MVA,
                              "MUFFIN": MUFFIN,
                              "DeepDDI": DeepDDI,
                              "DDKG": DDKG,
                              "SumGNN": SumGNN,
                              "KGNN": KGNN,
                              "LaGAT": LaGAT,
                              "PHGLDDI": PHGLDDI,
                              "MMDGDTI": MMDGDTI,
                              "ExDDI": ExDDI,
                              "MIRACLE": MIRACLE,
                              "CASTER": CASTER,
                               "MKGFENN": MKGFENN,
                              }

    def load_model(self):
        """
        Load the model corresponding to the specified model type.
        
        Returns:
            Initialized model instance on the appropriate device
            
        Raises:
            ValueError: If num_classes is not properly set
        """
        num_classes = int(getattr(self.args, 'num_classes', 0))
        if num_classes <= 0:
            raise ValueError("num_classes is not properly set; please assign value after data loading.")

        cls = self.model_mapping[self.args.model]
        want = set(signature(cls.__init__).parameters.keys())

        kwargs = {}

        # --- Dimensions: multi-modal list vs single dimension ---
        if 'features' in want:
            # Only pass list when model actually declares 'features' (like MKGFENN etc.)
            kwargs['features'] = self.args.features
        if 'feature' in want:
            # Most models only need the merged dimension
            kwargs['feature'] = int(self.args.dimensions)

        # --- Common parameters (pass if available) ---
        for k in ('hidden1', 'hidden2', 'dropout', 'num_classes',
                  'event_sem_dim', 'lambda_align', 'lambda_u_pair',
                  'lambda_u_event', 'uniform_t'):
            if k in want and hasattr(self.args, k):
                kwargs[k] = getattr(self.args, k)

        # Some models use num_relations (usually equal to number of classes)
        if 'num_relations' in want:
            kwargs['num_relations'] = num_classes

        model = cls(**kwargs)

        # Send to device
        device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model
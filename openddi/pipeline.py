# pipeline.py
import os
from inspect import signature
import torch

# Only used when probing dimensions from CSV
try:
    import pandas as pd
except Exception:
    pd = None

from models.model_manager import model_manager

from trainer.MRCGNN_Trainer import MRCGNN_Trainer
from trainer.ZeroDDI_Trainer import ZeroDDI_Trainer
from trainer.Unified_Trainer import Unified_Trainer


class Pipeline:
    """
    Pipeline class for managing model training workflow.
    
    Handles model initialization, trainer selection, and feature dimension validation.
    
    Args:
        args: Configuration arguments
        logger: Logger instance for logging
        dataset: Dataset object containing data loaders
        model: Neural network model to train
        optimizer: Optimizer for training
    """
    
    def __init__(self, args, logger, dataset, model, optimizer):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        # Key: Ensure args.features / args.dimensions are consistent with data; rebuild model/optimizer if necessary
        self._ensure_modal_dims_and_model()

        self.trainer_mapping = {
            "MRCGNN": MRCGNN_Trainer,
            "GOGNN": Unified_Trainer,        # Original mapping preserved
            "ZeroDDI": ZeroDDI_Trainer,
            "DDIMDL": Unified_Trainer,
            "ConvLSTM": Unified_Trainer,
            "MVA": Unified_Trainer,
            "MUFFIN": Unified_Trainer,
            "TIGER": Unified_Trainer,
            "DeepDDI": Unified_Trainer,
            "DDKG": Unified_Trainer,
            "SumGNN": Unified_Trainer,
            "KGNN": Unified_Trainer,
            "LaGAT": Unified_Trainer,
            "PHGLDDI": Unified_Trainer,
            "MMDGDTI": Unified_Trainer,
            "ExDDI": Unified_Trainer,
            "MIRACLE": Unified_Trainer,
            "CASTER": Unified_Trainer,
            "MKGFENN": Unified_Trainer,
        }
        self.trainer = self.load_trainer()

    def run(self):
        """
        Execute the main training pipeline.
        """
        if self.args.task == 'train_xxxx':
            self.trainer.train()

    def load_trainer(self):
        """
        Load the appropriate trainer based on model type.
        
        Returns:
            Trainer instance for the specified model
        """
        return self.trainer_mapping[self.args.model](self.args, self.logger, self.dataset, self.model, self.optimizer)

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _ensure_modal_dims_and_model(self):
        """
        1) Infer modal dimensions from dataset or embedding paths, write back to args.features / args.dimensions.
        2) If current model construction requires features and doesn't match args.features, rebuild model and optimizer.
        """
        # 1) Write back modal dimensions
        features = None

        # Priority: modal dimensions exposed by dataset (if stored during data loading)
        for attr in ["modal_dims", "feature_dims", "modality_dims"]:
            md = getattr(self.dataset, attr, None)
            if isinstance(md, (list, tuple)) and len(md) > 0:
                features = list(map(int, md))
                break

        # Secondary: probe dimensions from --embedding_path (only read first line/one sample, not too heavy)
        if features is None:
            paths = getattr(self.args, "embedding_path", None)
            if paths:
                features = self._probe_modal_dims_from_paths(paths)

        # If dimensions found, write back to args
        if features:
            self.args.features = list(map(int, features))
            self.args.dimensions = int(sum(self.args.features))
            if self.logger:
                self.logger.info(f"[Pipeline] modal features = {self.args.features} (sum={self.args.dimensions})")
            else:
                print(f"[Pipeline] modal features = {self.args.features} (sum={self.args.dimensions})")

        # 2) Check model construction signature, rebuild if necessary
        cls = type(self.model)
        want = set(signature(cls.__init__).parameters.keys())
        need_features = ("features" in want) and ("feature" not in want)  # e.g., DDIMDL type
        need_feature  = ("feature" in want)                                # e.g., most models

        # Is current instance consistent with args?
        ok = True
        if need_features:
            ok = hasattr(self.args, "features") and isinstance(self.args.features, (list, tuple)) and len(self.args.features) > 0
            # If model can get its features, further compare
            if ok and hasattr(self.model, "features"):
                try:
                    cur = list(getattr(self.model, "features"))
                    ok = list(map(int, cur)) == list(map(int, self.args.features))
                except Exception:
                    pass
        elif need_feature:
            ok = hasattr(self.args, "dimensions") and int(self.args.dimensions) > 0

        if not ok:
            if self.logger:
                self.logger.info("[Pipeline] Rebuilding model because input feature dims changed/missing.")
            else:
                print("[Pipeline] Rebuilding model because input feature dims changed/missing.")

            # Use model_manager to rebuild model with latest args
            man = model_manager(self.args)
            self.model = man.load_model()

            # Also rebuild optimizer (if you override later in main, this is fine)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(getattr(self.args, "lr", 1e-3)),
                weight_decay=float(getattr(self.args, "weight_decay", 5e-4)),
            )

    def _probe_modal_dims_from_paths(self, paths):
        """
        Quickly probe modal dimensions from embedding paths (.pt/.csv).
        
        Args:
            paths: List of file paths to embedding files
            
        Returns:
            list: List of dimension sizes for each modality
        """
        dims = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".pt":
                d = torch.load(p, map_location="cpu")
                # Expected format: {id: vector}
                k = next(iter(d.keys()))
                v = d[k]
                v = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                dims.append(int(v.shape[-1] if v.ndim > 1 else v.shape[0]))
            elif ext == ".csv":
                if pd is None:
                    raise RuntimeError("需要 pandas 来从 CSV 探测维度，请安装 pandas。")
                df = pd.read_csv(p, nrows=1)  # Only read 1 row to get column count
                dims.append(int(df.shape[1] - 1))  # Subtract first column (id)
            else:
                raise ValueError(f"不支持的嵌入文件后缀: {p}")
        return dims

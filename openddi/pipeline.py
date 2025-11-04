# pipeline.py
import os
from inspect import signature
import torch

# 仅在需要从 CSV 探测维度时用到
try:
    import pandas as pd
except Exception:
    pd = None

from models.model_manager import model_manager

from trainer.MRCGNN_Trainer import MRCGNN_Trainer
from trainer.ZeroDDI_Trainer import ZeroDDI_Trainer
from trainer.Unified_Trainer import Unified_Trainer



class Pipeline:
    def __init__(self, args, logger, dataset, model, optimizer):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        # 关键：确保 args.features / args.dimensions 与数据一致；必要时重建模型/优化器
        self._ensure_modal_dims_and_model()

        self.trainer_mapping = {
            "MRCGNN": MRCGNN_Trainer,
            "GOGNN": Unified_Trainer,        # 你原来的映射我保持不变
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
        if self.args.task == 'train_xxxx':
            self.trainer.train()

    def load_trainer(self):
        return self.trainer_mapping[self.args.model](self.args, self.logger, self.dataset, self.model, self.optimizer)

    # ---------------------------
    # 内部工具
    # ---------------------------
    def _ensure_modal_dims_and_model(self):
        """
        1) 尽量从 dataset 或 embedding 路径推断各模态维度，回写到 args.features / args.dimensions。
        2) 如果当前模型构造需要 features 且与 args.features 不一致，则重建模型和优化器。
        """
        # 1) 回写模态维度
        features = None

        # 优先：dataset 暴露的模态维度（如果你在数据加载里存了的话）
        for attr in ["modal_dims", "feature_dims", "modality_dims"]:
            md = getattr(self.dataset, attr, None)
            if isinstance(md, (list, tuple)) and len(md) > 0:
                features = list(map(int, md))
                break

        # 其次：根据 --embedding_path 探维度（只读第一行/一个样本，不会太重）
        if features is None:
            paths = getattr(self.args, "embedding_path", None)
            if paths:
                features = self._probe_modal_dims_from_paths(paths)

        # 如果探到了，回写 args
        if features:
            self.args.features = list(map(int, features))
            self.args.dimensions = int(sum(self.args.features))
            if self.logger:
                self.logger.info(f"[Pipeline] modal features = {self.args.features} (sum={self.args.dimensions})")
            else:
                print(f"[Pipeline] modal features = {self.args.features} (sum={self.args.dimensions})")

        # 2) 检查模型构造签名，必要时重建
        cls = type(self.model)
        want = set(signature(cls.__init__).parameters.keys())
        need_features = ("features" in want) and ("feature" not in want)  # 例如 DDIMDL 这类
        need_feature  = ("feature" in want)                                # 例如大多数模型

        # 当前实例是否与 args 一致？
        ok = True
        if need_features:
            ok = hasattr(self.args, "features") and isinstance(self.args.features, (list, tuple)) and len(self.args.features) > 0
            # 如果模型里能拿到它的 features，也进一步比对一下
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

            # 用 model_manager 按最新 args 重建模型
            man = model_manager(self.args)
            self.model = man.load_model()

            # 也重建优化器（若你在 main 里之后还会覆盖，这里也没问题）
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(getattr(self.args, "lr", 1e-3)),
                weight_decay=float(getattr(self.args, "weight_decay", 5e-4)),
            )

    def _probe_modal_dims_from_paths(self, paths):
        """从 embedding 路径快速探测各模态维度（.pt/.csv）。"""
        dims = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".pt":
                d = torch.load(p, map_location="cpu")
                # 期望是 {id: vector}
                k = next(iter(d.keys()))
                v = d[k]
                v = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                dims.append(int(v.shape[-1] if v.ndim > 1 else v.shape[0]))
            elif ext == ".csv":
                if pd is None:
                    raise RuntimeError("需要 pandas 来从 CSV 探测维度，请安装 pandas。")
                df = pd.read_csv(p, nrows=1)  # 只读 1 行足够知道列数
                dims.append(int(df.shape[1] - 1))  # 去掉第一列 id
            else:
                raise ValueError(f"不支持的嵌入文件后缀: {p}")
        return dims

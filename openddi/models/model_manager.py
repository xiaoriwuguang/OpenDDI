import argparse
import torch
from models.gognn.gognn import GoGNN as origognn
from models.muffin.muffin import MUFFIN as orimuffin
from models.mva.mva import MVA as orimva
from models.MRCGNN import MRCGNN
from models.GOGNN import GOGNN
from models.ZeroDDI import ZeroDDI
from models.DDIMDL import DDIMDL    
from models.TIGER import TIGER
from models.tiger.tiger import TIGER as oritiger
from models.ConvLSTM import ConvLSTM
from models.MVA import MVA
from models.MUFFIN import MUFFIN
from models.DeepDDI import DeepDDI
from models.DDKG import DDKG
from models.SumGNN import SumGNN
from models.KGNN import KGNN
from models.DSNDDI import DSNDDI
from models.LaGAT import LaGAT
from models.PHGLDDI import PHGLDDI
from models.MMDGDTI import MMDGDTI 
from models.ExDDI import ExDDI
from models.MIRACLE import MIRACLE
from models.CASTER import CASTER 
from models.MKGFENN import MKGFENN
from inspect import signature

class model_manager:
    def __init__(self,
                 args:argparse):
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
                              "DSNDDI": DSNDDI,
                              "ExDDI": ExDDI,
                              "MIRACLE": MIRACLE,
                              "CASTER": CASTER,
                              "MKGFENN": MKGFENN,
                              }

    def load_model(self):
        num_classes = int(getattr(self.args, 'num_classes', 0))
        if num_classes <= 0:
            raise ValueError("num_classes 未正确设置；请先在数据加载后赋值。")

        cls = self.model_mapping[self.args.model]
        want = set(signature(cls.__init__).parameters.keys())

        kwargs = {}

        # --- 维度：多模态列表 vs 单一维度 ---
        if 'features' in want:
            # 仅在模型真的声明了 'features' 时才传列表（如 MKGFENN 等）
            kwargs['features'] = self.args.features
        if 'feature' in want:
            # 大多数模型只要合并后的维度
            kwargs['feature'] = int(self.args.dimensions)

        # --- 常见参数（有就传）---
        for k in ('hidden1', 'hidden2', 'dropout', 'num_classes',
                'event_sem_dim', 'lambda_align', 'lambda_u_pair',
                'lambda_u_event', 'uniform_t'):
            if k in want and hasattr(self.args, k):
                kwargs[k] = getattr(self.args, k)

        # 有些模型叫 num_relations（通常等于类别数）
        if 'num_relations' in want:
            kwargs['num_relations'] = num_classes

        model = cls(**kwargs)

        # 发送到设备
        device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model
    
    def load_origin_model(self, ddi_dataset):
        num_classes = int(getattr(self.args, 'num_classes', 0))
        device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if num_classes <= 0:
            raise ValueError("num_classes 未正确设置；请先在数据加载后赋值。")
        if self.args.model == "TIGER":
            model = oritiger(max_layer = 2,
                        num_features_drug = 67,
                        num_nodes = ddi_dataset.data_sta['num_nodes'],
                        num_relations_mol = ddi_dataset.data_sta['num_rel_mol'],
                        num_relations_graph = ddi_dataset.data_sta['num_rel_graph'],
                        output_dim=64,
                        max_degree_graph=ddi_dataset.data_sta['max_degree_graph'],
                        max_degree_node=ddi_dataset.data_sta['max_degree_node'],
                        sub_coeff = 0.2,
                        mi_coeff = 0.5,
                        dropout=0.2,
                        device = device,
                        num_rel = num_classes,
                        args=self.args)
        if self.args.model == "GOGNN":
            model = origognn(args = self.args, 
                        num_features = ddi_dataset.num_features, 
                        nhid = 64, 
                        ddi_nhid = 64, 
                        pooling_ratio = 0.3, 
                        dropout_ratio = 0.3,
                        num_rel = num_classes)
        if self.args.model == "MUFFIN":
            model = orimuffin(args = self.args, 
                        entity_dim = ddi_dataset.entity_dim, 
                        structure_dim = ddi_dataset.structure_dim, 
                        num_rel = num_classes)
        if self.args.model == "MVA":
            model = orimva(args = self.args, 
                        gcn_in_features = 75, 
                        gcn_out_features = 128, 
                        num_rel = num_classes)

        # 发送到设备
        model.to(device)
        return model

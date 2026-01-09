import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_INIT_EPS = 2.0
gamma = 12.0


class MUFFIN(nn.Module):

    def __init__(self, args, entity_dim, structure_dim, num_rel):

        super(MUFFIN, self).__init__()
        self.args = args

        self.use_pretrain = 1
        self.entity_dim = entity_dim
        self.structure_dim = structure_dim
        self.fusion_type = 'init_double'


        self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
        self.druglayer_KG = nn.Linear(self.entity_dim, self.entity_dim)
        self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
        self.activate = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(22 * 22 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                     nn.ReLU(True))

        self.fc2_global = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))
        self.fc2_global_reverse = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))
        self.fc2_cross = nn.Sequential(
            nn.Linear(self.entity_dim * 4, self.entity_dim),
            nn.ReLU(True))

        if self.fusion_type in ['double', 'init_double']:
            self.all_embedding_dim = (self.entity_dim * 3 + self.structure_dim + self.entity_dim) * 2

        self.layer1 = nn.Sequential(nn.Linear(self.all_embedding_dim, 2048), nn.BatchNorm1d(2048),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(2048, num_rel))

    def generate_fusion_feature(self, batch_data):
        # we focus on approved drug
        entity_embed_pre = batch_data[0]
        structure_embed_pre = batch_data[1]

        if self.fusion_type == 'init_double':

            structure = self.druglayer_structure(structure_embed_pre)

            entity = self.druglayer_KG(entity_embed_pre)

            structure_embed_reshape = structure.unsqueeze(-1)  # batch_size * embed_dim * 1
            entity_embed_reshape = entity.unsqueeze(-1)  # batch_size * embed_dim * 1

            entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_global = entity_matrix.view(entity_matrix.size(0), -1)

            entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

            entity_matrix_reshape = entity_matrix.unsqueeze(1)

            # Direct processing without loop
            out = self.conv1(entity_matrix_reshape)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            embedding_data = self.fc1(out)

            global_local_before = torch.cat((embedding_data, entity_global), 1)
            cross_embedding_pre = self.fc2_global(global_local_before)

            # another reverse part

            entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)

            # Direct processing without loop
            out = self.conv1(entity_matrix_reshape_reverse)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            embedding_data_reverse = self.fc1(out)

            global_local_before_reverse = torch.cat((embedding_data_reverse, entity_global_reverse), 1)
            cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)

            out3 = self.activate(self.multi_drug(structure * entity))

            out_concat = torch.cat(
                (structure_embed_pre, entity_embed_pre, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)

            return out_concat

    def forward(self, batch_data):

        all_embed = self.generate_fusion_feature(batch_data)
        ddi_edge_index = batch_data[2]
        source, target = ddi_edge_index

        drug1_embed = all_embed[source]
        drug2_embed = all_embed[target]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    def loss(self, logits, labels):
        """Supervised loss for DDI edge classification.

        If args.matrix in ['multilabel','twosides'] -> BCEWithLogitsLoss (labels float multi-hot)
        else -> CrossEntropyLoss (labels long class indices)
        """
        task = getattr(self.args, 'matrix', 'multiclass')
        if task in ['multilabel', 'twosides']:
            labels = labels.float()
            return nn.BCEWithLogitsLoss()(logits, labels)
        else:
            return nn.CrossEntropyLoss()(logits, labels.long())


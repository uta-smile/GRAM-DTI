import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import GRAM4ModalLoss


class OptimizedFourModalContrastiveModel(nn.Module):

    def __init__(self, temperature=0.07, projection_dim=512, class_weights=None, drop_prob=0.3,
                 lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, gradient_std_multiplier=1.5, gradient_history_length=5):
        super().__init__()
        self.temperature = temperature

        self.gram_loss = GRAM4ModalLoss(
            contra_temp=temperature,
            lambda_dam=0.5,
            lambda_ic50=0.5,
            class_weights=class_weights,
            drop_prob=drop_prob,
            projection_dim=projection_dim,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            gradient_std_multiplier=gradient_std_multiplier,
            gradient_history_length=gradient_history_length
        )

        self.smiles_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim)
        )

        self.protein_projector = nn.Sequential(
            nn.Linear(1280, 768), 
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim)
        )

        self.ic50_classifier = nn.Sequential(
            nn.Linear(projection_dim * 4, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def project_embeddings(self, smiles_emb, text_emb, hta_emb, protein_emb):
        smiles_projected = nn.functional.normalize(self.smiles_projector(smiles_emb), dim=1)
        text_projected = nn.functional.normalize(self.text_projector(text_emb), dim=1)
        hta_projected = nn.functional.normalize(self.text_projector(hta_emb), dim=1)
        protein_projected = nn.functional.normalize(self.protein_projector(protein_emb), dim=1)

        return smiles_projected, text_projected, hta_projected, protein_projected

    def predict_ic50_class(self, smiles_feat, text_feat, hta_feat, protein_feat):
        combined_feat = torch.cat([smiles_feat, text_feat, hta_feat, protein_feat], dim=1)
        ic50_logits = self.ic50_classifier(combined_feat)
        return ic50_logits

    def forward(self, smiles_emb, text_emb, hta_emb, protein_emb, ic50_targets=None, ic50_mask=None):
        smiles_projected, text_projected, hta_projected, protein_projected = self.project_embeddings(
            smiles_emb, text_emb, hta_emb, protein_emb
        )
        ic50_logits = self.predict_ic50_class(
            smiles_projected, text_projected, hta_projected, protein_projected
        )

        total_loss, contra_loss, itm_loss, ic50_loss, ic50_acc, drop_info = self.gram_loss(
            text_feats=text_projected,
            smiles_feats=smiles_projected,
            hta_feats=hta_projected,
            protein_feats=protein_projected,
            ic50_logits=ic50_logits,
            ic50_targets=ic50_targets,
            ic50_mask=ic50_mask
        )

        return total_loss, contra_loss, itm_loss, ic50_loss, ic50_acc, ic50_logits, drop_info



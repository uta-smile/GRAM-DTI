import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class GRAM4ModalLoss(nn.Module):
    def __init__(self, contra_temp=0.07, lambda_dam=0.5, lambda_ic50=1.0, class_weights=None, drop_prob=0.3, projection_dim=512,
                 lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, gradient_std_multiplier=1.5, gradient_history_length=5):
        super().__init__()
        self.contra_temp = contra_temp
        self.lambda_dam = lambda_dam
        self.lambda_ic50 = lambda_ic50
        self.drop_prob = drop_prob

        # Loss weights for hyperparameter control
        self.lambda_1 = lambda_1  
        self.lambda_2 = lambda_2  
        self.lambda_3 = lambda_3  

        self.gradient_std_multiplier = gradient_std_multiplier
        self.gradient_history_length = gradient_history_length

        # Register class weights for handling class imbalance
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.gradient_history = {
            'text': [],
            'smiles': [],
            'hta': [],
            'protein': []
        }
        self.gradient_drop_strategy = 'adaptive'  # 'high', 'low', 'adaptive'

        self.gradient_weight_decay = 0.9

    def compute_4modal_volume_batch(self, anchor_feats, feat1_all, feat2_all, feat3_all):
        # Use features directly without weights
        anchor_feats_weighted = anchor_feats
        feat1_weighted = feat1_all
        feat2_weighted = feat2_all
        feat3_weighted = feat3_all

        batch_size1 = anchor_feats_weighted.shape[0]
        batch_size2 = feat1_weighted.shape[0]

        # Compute pairwise dot products for 4x4 Gram matrix
        # anchor vs anchor (diagonal terms)
        aa = torch.einsum('bi,bi->b', anchor_feats_weighted, anchor_feats_weighted).unsqueeze(1).expand(-1, batch_size2)
        # anchor vs feat1
        af1 = anchor_feats_weighted @ feat1_weighted.T
        # anchor vs feat2
        af2 = anchor_feats_weighted @ feat2_weighted.T
        # anchor vs feat3
        af3 = anchor_feats_weighted @ feat3_weighted.T

        # feat1 vs feat1 (diagonal terms)
        f1f1 = torch.einsum('bi,bi->b', feat1_weighted, feat1_weighted).unsqueeze(0).expand(batch_size1, -1)
        # feat1 vs feat2
        f1f2 = torch.einsum('bi,bi->b', feat1_weighted, feat2_weighted).unsqueeze(0).expand(batch_size1, -1)
        # feat1 vs feat3
        f1f3 = torch.einsum('bi,bi->b', feat1_weighted, feat3_weighted).unsqueeze(0).expand(batch_size1, -1)

        # feat2 vs feat2 (diagonal terms)
        f2f2 = torch.einsum('bi,bi->b', feat2_weighted, feat2_weighted).unsqueeze(0).expand(batch_size1, -1)
        # feat2 vs feat3
        f2f3 = torch.einsum('bi,bi->b', feat2_weighted, feat3_weighted).unsqueeze(0).expand(batch_size1, -1)

        # feat3 vs feat3 (diagonal terms)
        f3f3 = torch.einsum('bi,bi->b', feat3_weighted, feat3_weighted).unsqueeze(0).expand(batch_size1, -1)

        # Stack to form 4x4 Gram matrix
        G = torch.stack([
            torch.stack([aa, af1, af2, af3], dim=-1),
            torch.stack([af1, f1f1, f1f2, f1f3], dim=-1),
            torch.stack([af2, f1f2, f2f2, f2f3], dim=-1),
            torch.stack([af3, f1f3, f2f3, f3f3], dim=-1)
        ], dim=-2)

        # Compute determinant and volume
        gram_det = torch.det(G.float())
        volumes = torch.sqrt(torch.abs(gram_det))

        return volumes

    def compute_3modal_volume_batch(self, anchor_feats, feat1_all, feat2_all):
        anchor_feats_weighted = anchor_feats
        feat1_weighted = feat1_all
        feat2_weighted = feat2_all

        batch_size1 = anchor_feats_weighted.shape[0]
        batch_size2 = feat1_weighted.shape[0]

        # Compute pairwise dot products for 3x3 Gram matrix
        # anchor vs anchor (diagonal terms)
        aa = torch.einsum('bi,bi->b', anchor_feats_weighted, anchor_feats_weighted).unsqueeze(1).expand(-1, batch_size2)
        # anchor vs feat1
        af1 = anchor_feats_weighted @ feat1_weighted.T
        # anchor vs feat2
        af2 = anchor_feats_weighted @ feat2_weighted.T

        # feat1 vs feat1 (diagonal terms)
        f1f1 = torch.einsum('bi,bi->b', feat1_weighted, feat1_weighted).unsqueeze(0).expand(batch_size1, -1)
        # feat1 vs feat2
        f1f2 = torch.einsum('bi,bi->b', feat1_weighted, feat2_weighted).unsqueeze(0).expand(batch_size1, -1)

        # feat2 vs feat2 (diagonal terms)
        f2f2 = torch.einsum('bi,bi->b', feat2_weighted, feat2_weighted).unsqueeze(0).expand(batch_size1, -1)

        # Stack to form 3x3 Gram matrix
        G = torch.stack([
            torch.stack([aa, af1, af2], dim=-1),
            torch.stack([af1, f1f1, f1f2], dim=-1),
            torch.stack([af2, f1f2, f2f2], dim=-1)
        ], dim=-2)

        # Compute determinant and volume
        gram_det = torch.det(G.float())
        volumes = torch.sqrt(torch.abs(gram_det))

        return volumes

    def calculate_modality_gradients(self, text_feats, smiles_feats, hta_feats, protein_feats,
                                   current_loss):
        if not self.training:
            return {'text': 0.0, 'smiles': 0.0, 'hta': 0.0, 'protein': 0.0}

        gradient_norms = {}

        modality_features = {
            'text': text_feats,
            'smiles': smiles_feats,
            'hta': hta_feats,
            'protein': protein_feats
        }

        for modality_name, features in modality_features.items():
            if not features.requires_grad:
                features.requires_grad_(True)

            try:
                grad = torch.autograd.grad(
                    outputs=current_loss,
                    inputs=features,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )[0]

                if grad is not None:
                    gradient_norms[modality_name] = torch.norm(grad, p=2).item()
                else:
                    gradient_norms[modality_name] = 0.0

            except (RuntimeError, TypeError):
                gradient_norms[modality_name] = 0.0

        return gradient_norms

    def update_gradient_history(self, gradient_norms):
        for modality in ['text', 'smiles', 'hta', 'protein']:
            self.gradient_history[modality].append(gradient_norms.get(modality, 0.0))

            if len(self.gradient_history[modality]) > self.gradient_history_length:
                self.gradient_history[modality].pop(0)

    def get_gradient_based_drop_decision(self):
        if not self.training:
            return 'none', {}

        min_history = min(len(self.gradient_history[mod]) for mod in ['text', 'smiles', 'hta', 'protein'])
        if min_history < 2:
            modalities = ['none', 'text', 'smiles', 'hta', 'protein']
            return random.choice(modalities), {}

        weighted_gradients = {}
        for modality in ['text', 'smiles', 'hta', 'protein']:
            history = self.gradient_history[modality]
            weights = [self.gradient_weight_decay ** (len(history) - 1 - i) for i in range(len(history))]
            weighted_avg = sum(h * w for h, w in zip(history, weights)) / sum(weights)
            weighted_gradients[modality] = weighted_avg

        if self.gradient_drop_strategy == 'high':
            drop_modality = max(weighted_gradients.keys(), key=lambda k: weighted_gradients[k])
        elif self.gradient_drop_strategy == 'low':
            min_grad = min(weighted_gradients.values())
            min_candidates = [modality for modality, grad in weighted_gradients.items() if grad == min_grad]
            drop_modality = random.choice(min_candidates)
        else:
            gradient_mean = np.mean(list(weighted_gradients.values()))
            gradient_std = np.std(list(weighted_gradients.values()))
            for modality, grad in weighted_gradients.items():
                if grad > gradient_mean + self.gradient_std_multiplier * gradient_std:
                    drop_modality = modality
                    break
            else:
                min_grad = min(weighted_gradients.values())
                min_candidates = [modality for modality, grad in weighted_gradients.items() if grad == min_grad]
                drop_modality = random.choice(min_candidates)
        return drop_modality, weighted_gradients

    def decide_drop_strategy_once(self, text_feats, smiles_feats, hta_feats, protein_feats, current_loss):

        if not self.training or random.random() > self.drop_prob:
            return {
                "should_drop": False,
                "dropped_modality": "none",
                "anchor_modality": "protein",
                "remaining_modalities": ["text", "smiles", "hta", "protein"],
                "gradient_scores": {},
                "current_gradients": {}
            }

        gradient_norms = self.calculate_modality_gradients(
            text_feats, smiles_feats, hta_feats, protein_feats, current_loss
        )

        self.update_gradient_history(gradient_norms)

        dropped_modality, gradient_scores = self.get_gradient_based_drop_decision()

        if dropped_modality == 'none':
            return {
                "should_drop": False,
                "dropped_modality": "none",
                "anchor_modality": "protein",
                "remaining_modalities": ["text", "smiles", "hta", "protein"],
                "gradient_scores": gradient_scores,
                "current_gradients": gradient_norms
            }

        all_modalities = ['text', 'hta', 'smiles', 'protein']
        remaining_modalities = [m for m in all_modalities if m != dropped_modality]

        anchor_modality = random.choice(remaining_modalities)

        return {
            "should_drop": True,
            "dropped_modality": dropped_modality,
            "anchor_modality": anchor_modality,
            "remaining_modalities": remaining_modalities,
            "gradient_scores": gradient_scores,
            "current_gradients": gradient_norms
        }

    def compute_volumes_with_consistent_drop(self, text_feats, smiles_feats, hta_feats, protein_feats,
                                           text_all, smiles_all, hta_all, protein_all, drop_strategy):
        modality_feats = {
            'text': text_feats,
            'smiles': smiles_feats,
            'hta': hta_feats,
            'protein': protein_feats
        }

        modality_all = {
            'text': text_all,
            'smiles': smiles_all,
            'hta': hta_all,
            'protein': protein_all
        }

        if not drop_strategy["should_drop"]:
            volume_forward = self.compute_4modal_volume_batch(
                protein_feats, text_all, hta_all, smiles_all
            )
            volume_reverse = self.compute_4modal_volume_batch(
                protein_all, text_feats, hta_feats, smiles_feats
            ).T
        else:
            dropped = drop_strategy["dropped_modality"]
            anchor = drop_strategy["anchor_modality"]
            remaining = drop_strategy["remaining_modalities"]

            non_anchor_modalities = [m for m in remaining if m != anchor]

            anchor_feats = modality_feats[anchor]
            feat1_all = modality_all[non_anchor_modalities[0]]
            feat2_all = modality_all[non_anchor_modalities[1]]
            volume_forward = self.compute_3modal_volume_batch(anchor_feats, feat1_all, feat2_all)

            anchor_all = modality_all[anchor]
            feat1_feats = modality_feats[non_anchor_modalities[0]]
            feat2_feats = modality_feats[non_anchor_modalities[1]]
            volume_reverse = self.compute_3modal_volume_batch(anchor_all, feat1_feats, feat2_feats).T

        return volume_forward, volume_reverse

    def volume_to_similarity(self, volumes, method="negative"):
        if method == "negative":
            return -volumes
        elif method == "inverse":
            epsilon = 1e-8
            return 1.0 / (volumes + epsilon)
        elif method == "negative_log":
            epsilon = 1e-8
            return -torch.log(volumes + epsilon)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, text_feats, smiles_feats, hta_feats, protein_feats,
                ic50_logits, ic50_targets, ic50_mask):
        text_all = text_feats
        smiles_all = smiles_feats
        hta_all = hta_feats
        protein_all = protein_feats

        B = text_feats.size(0)
        targets = torch.arange(B, device=text_feats.device)

        ic50_loss = 0.0
        ic50_acc = 0.0
        baseline_loss_type = 'contrastive'

        if ic50_mask.sum() > 0:
            valid_mask = ic50_mask > 0
            if valid_mask.sum() > 0:
                ic50_logits_valid = ic50_logits[valid_mask]
                ic50_targets_valid = ic50_targets[valid_mask]

                if self.class_weights is not None:
                    ic50_loss = F.cross_entropy(ic50_logits_valid, ic50_targets_valid,
                                              weight=self.class_weights, label_smoothing=0.1)
                else:
                    ic50_loss = F.cross_entropy(ic50_logits_valid, ic50_targets_valid, label_smoothing=0.1)

                ic50_pred = torch.argmax(ic50_logits_valid, dim=1)
                ic50_acc = (ic50_pred == ic50_targets_valid).float().mean()
                baseline_loss_type = 'ic50'

        sim_s2p = (smiles_feats) @ (protein_all).T / self.contra_temp
        sim_p2s = (protein_feats) @ (smiles_all).T / self.contra_temp

        loss_s2p = F.cross_entropy(sim_s2p, targets, label_smoothing=0.1)
        loss_p2s = F.cross_entropy(sim_p2s, targets, label_smoothing=0.1)
        itm_loss = (loss_s2p + loss_p2s) / 2

        if isinstance(ic50_loss, torch.Tensor) and ic50_loss.item() != 0.0:
            baseline_loss = self.lambda_2 * itm_loss + self.lambda_3 * ic50_loss
        else:
            baseline_loss = self.lambda_2 * itm_loss



        drop_strategy = self.decide_drop_strategy_once(
            text_feats, smiles_feats, hta_feats, protein_feats, baseline_loss
        )

        volume_forward, volume_reverse = self.compute_volumes_with_consistent_drop(
            text_feats, smiles_feats, hta_feats, protein_feats,
            text_all, smiles_all, hta_all, protein_all,
            drop_strategy
        )

        similarity_forward = self.volume_to_similarity(volume_forward, method="negative") / self.contra_temp
        similarity_reverse = self.volume_to_similarity(volume_reverse, method="negative") / self.contra_temp

        loss_forward = F.cross_entropy(similarity_forward, targets, label_smoothing=0.1)
        loss_reverse = F.cross_entropy(similarity_reverse, targets, label_smoothing=0.1)

        contra_loss = (loss_forward + loss_reverse) / 2

        total_loss = self.lambda_1 * contra_loss + self.lambda_2 * itm_loss + self.lambda_3 * ic50_loss

        drop_info = {
            'strategy': drop_strategy,
            'baseline_loss_type': baseline_loss_type
        }

        return total_loss, contra_loss, itm_loss, ic50_loss, ic50_acc, drop_info



"""Evaluator: Performance prediction model for analog circuits.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module implements an evaluator that predicts circuit performance specifications
(gain, bandwidth, phase margin, figure-of-merit) from circuit topology using
contrastive learning and multi-task prediction.
"""

import torch
import torch.nn as nn

from models.tools.losses import InfoNCE_with_filtering

  
class Spec_Embedder(nn.Module):
    """Embeds discretized specifications into continuous latent space.
    
    Attributes:
        gain_embed: Embedding layer for gain bins.
        pm_embed: Embedding layer for phase margin bins.
        bw_embed: Embedding layer for bandwidth bins.
        cond_proj: Projection layer combining all embeddings.
        fc_embs: Final linear layer to produce specification embeddings.
    """
    
    def __init__(self, num_gain_type, num_pm_type, num_bw_type, emb_dim=128, latent_dim=64):
        """Initializes the specification embedder.
        
        Args:
            num_gain_type: Number of discretized gain bins.
            num_pm_type: Number of discretized phase margin bins.
            num_bw_type: Number of discretized bandwidth bins.
            emb_dim: Embedding dimension for each spec type.
            latent_dim: Output latent dimension.
        """
        super().__init__()

        self.gain_embed = nn.Embedding(num_gain_type, emb_dim)
        self.pm_embed   = nn.Embedding(num_pm_type, emb_dim)
        self.bw_embed   = nn.Embedding(num_bw_type, emb_dim)

        self.cond_proj = nn.Linear(emb_dim*3, emb_dim)
        self.fc_embs  = nn.Linear(emb_dim, latent_dim)
        
    def forward(self, batch):
        """Embeds specifications into latent space.
        
        Args:
            batch: Dictionary with keys 'gains', 'bws', 'pms' (discretized indices).
            
        Returns:
            Specification embeddings of shape [batch_size, latent_dim].
        """
        gains, bws, pms = batch['gains'], batch['bws'], batch['pms']
        gain_embs = self.gain_embed(gains)
        bw_embs = self.bw_embed(bws)
        pm_embs = self.pm_embed(pms)

        # Concatenate and project
        conditions = torch.cat([gain_embs, bw_embs, pm_embs], dim=-1)
        conditions = self.cond_proj(conditions)

        spec_embs = self.fc_embs(conditions)
        return spec_embs


class EVALUATOR(nn.Module):
    """Performance evaluator for predicting circuit specifications.
    
    This model learns to predict circuit performance metrics (gain, bandwidth,
    phase margin, FOM) from circuit topology using a combination of contrastive
    learning (InfoNCE) and supervised prediction losses.
    
    Attributes:
        archi: Circuit encoder architecture.
        spec_embedder: Embeds specifications for contrastive learning.
        fc_gain, fc_pm, fc_bw: Classifiers for discretized specs.
        fc_fom: Regression head for figure-of-merit.
    """
    
    def __init__(self, archi, emb_dim, latent_dim, num_gain_type, num_pm_type, num_bw_type,
                 temperature, device, dropout=0.1, **kwargs):
        """Initializes the evaluator model.
        
        Args:
            archi: Circuit encoder architecture.
            emb_dim: Embedding dimension.
            latent_dim: Latent space dimension.
            num_gain_type: Number of gain bins for classification.
            num_pm_type: Number of phase margin bins.
            num_bw_type: Number of bandwidth bins.
            temperature: Temperature for InfoNCE loss.
            device: Torch device (cuda/cpu).
            dropout: Dropout rate (currently not used).
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.archi = archi
        self.spec_embedder = Spec_Embedder(
            num_gain_type=num_gain_type,  
            num_pm_type=num_pm_type,    
            num_bw_type=num_bw_type,    
            emb_dim=emb_dim,
            latent_dim=latent_dim
        )
        self.device = device

        # Statistics for FOM denormalization
        self.fom_train_mean = None
        self.fom_train_std = None

        # Specification classifiers
        self.fc_gain = nn.Sequential(nn.Linear(latent_dim, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, num_gain_type))

        self.fc_pm   = nn.Sequential(nn.Linear(latent_dim, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, num_pm_type))
                                        
        self.fc_bw   = nn.Sequential(nn.Linear(latent_dim, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, num_bw_type))

        # FOM regression
        self.fc_fom = nn.Sequential(nn.Linear(latent_dim, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1))
        
        # Loss functions
        self.align_loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
        self.infonce_loss_fn = InfoNCE_with_filtering(temperature=temperature)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)


    def predict(self, ckt_embs, topk=1):
        """Predicts specifications from circuit embeddings with denormalization.
        
        Args:
            ckt_embs: Circuit embeddings of shape [batch_size, latent_dim].
            topk: Number of top predictions to return for each spec.
            
        Returns:
            Dictionary with keys 'gain', 'bw', 'pm' (top-k class indices) and
            'fom' (denormalized regression values).
        """
        logits_gain = self.fc_gain(ckt_embs)
        logits_bw   = self.fc_bw(ckt_embs)
        logits_pm   = self.fc_pm(ckt_embs)

        _, pred_gain = torch.topk(logits_gain, k=topk, dim=-1)
        _, pred_bw   = torch.topk(logits_bw, k=topk, dim=-1)
        _, pred_pm   = torch.topk(logits_pm, k=topk, dim=-1)
        pred_fom   = self.fc_fom(ckt_embs) * self.fom_train_mean + self.fom_train_std

        return {'gain': pred_gain, 'bw': pred_bw, 'pm': pred_pm, 'fom': pred_fom}


    def predict_standard(self, ckt_embs):
        """Predicts specifications without denormalization (for training).
        
        Args:
            ckt_embs: Circuit embeddings of shape [batch_size, latent_dim].
            
        Returns:
            Dictionary with logits/normalized predictions for all specs.
        """
        pred_gain = self.fc_gain(ckt_embs).squeeze()
        pred_bw   = self.fc_bw(ckt_embs).squeeze()
        pred_pm   = self.fc_pm(ckt_embs).squeeze()
        pred_fom   = self.fc_fom(ckt_embs).squeeze()

        return {'gain': pred_gain, 'bw': pred_bw, 'pm': pred_pm, 'fom': pred_fom}


    def set_train_mean_std(self, mean, std):
        """Sets FOM normalization statistics from training data.
        
        Args:
            mean: Training set mean for FOM.
            std: Training set standard deviation for FOM.
        """
        self.fom_train_mean = mean
        self.fom_train_std = std


    def get_ckt_embeddings(self, batch):
        """Encodes circuits into embeddings.
        
        Args:
            batch: Dictionary containing circuit data.
            
        Returns:
            Circuit embeddings of shape [batch_size, latent_dim].
        """
        return self.archi(batch)

        
    def get_spec_embeddings(self, batch):
        """Encodes specifications into embeddings.
        
        Args:
            batch: Dictionary containing specification data.
            
        Returns:
            Specification embeddings of shape [batch_size, latent_dim].
        """
        return self.spec_embedder(batch)




    def compute_loss(self, batch):
        """Computes multi-task loss for evaluator training.
        
        Combines three loss types:
        1. Alignment loss: Aligns circuit and spec embeddings.
        2. Contrastive loss: InfoNCE with false negative filtering.
        3. Prediction loss: Classification (gain/BW/PM) + regression (FOM).
        
        Args:
            batch: Dictionary containing:
                - 'ckt_embs': Circuit embeddings.
                - 'spec_embs': Specification embeddings.
                - 'gains', 'bws', 'pms', 'foms': Ground truth labels.
                - 'filter_mask': Mask for filtering false negatives.
                
        Returns:
            Tuple of (total_loss, loss_dict) with individual loss components.
        """
        losses = {}
        mixed_loss = 0

        ckt_embs = batch['ckt_embs']
        spec_embs = batch['spec_embs'] 

        # Contrastive and alignment losses
        losses['align'] = self.align_loss_fn(spec_embs, ckt_embs)
        losses['nce'] = self.infonce_loss_fn(spec_embs, ckt_embs, mask=batch['filter_mask'])
        mixed_loss += 0.01*losses['align']
        mixed_loss += losses['nce']

        # Prediction losses
        gnd_gains, gnd_bws, gnd_pms, gnd_foms = batch['gains'], batch['bws'], batch['pms'], batch['foms']
        preds = self.predict_standard(ckt_embs)

        loss_gain     = self.ce_loss_fn(preds['gain'], gnd_gains)
        loss_bw       = self.ce_loss_fn(preds['bw'], gnd_bws)
        loss_pm       = self.ce_loss_fn(preds['pm'], gnd_pms)
        loss_fom      = self.regression_loss_fn(preds['fom'], gnd_foms)

        losses['pred'] = loss_gain + loss_bw + loss_pm + loss_fom
        mixed_loss += losses['pred']

        return mixed_loss, losses
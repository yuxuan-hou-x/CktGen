"""Specification encoder for performance metrics embedding.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
from torch import nn


class Spec_Encoder(nn.Module):
    """Encoder for circuit performance specifications.
    
    Encodes performance specifications (gain, phase margin, bandwidth) into
    latent representations that can be used to condition circuit generation.
    
    Args:
        device: Device to run the model on (CPU/GPU).
        num_gain_type: Number of discretized gain levels.
        num_pm_type: Number of discretized phase margin levels.
        num_bw_type: Number of discretized bandwidth levels.
        emb_dim: Embedding dimension for specifications.
        latent_dim: Dimension of output latent representation.
        vae: Whether to use VAE mode (outputs mu and logvar).
        **kwargs: Additional arguments.
    """
    def __init__(self, device, num_gain_type, num_pm_type, num_bw_type, emb_dim=128, latent_dim=64, vae=False, **kwargs):
        
        super().__init__()
        
        self.device     = device
        self.vae        = vae


        ##### ---- condition embedding ---- #####
        self.gain_embed = nn.Embedding(num_gain_type, emb_dim)
        self.pm_embed   = nn.Embedding(num_pm_type, emb_dim)
        self.bw_embed   = nn.Embedding(num_bw_type, emb_dim)

        ##### ---- map graph embed to latent ---- #####
        self.cond_proj = nn.Linear(emb_dim*3, emb_dim)

        if self.vae:
            self.fc_mu      = nn.Linear(emb_dim, latent_dim)
            self.fc_logvar  = nn.Linear(emb_dim, latent_dim)
        else:
            self.fc_latent  = nn.Linear(emb_dim, latent_dim)


    def forward(self, batch):
        """Encodes performance specifications into latent representations.
        
        Args:
            batch: Dictionary containing:
                - gains: Gain specification indices (batch_size,).
                - bws: Bandwidth specification indices (batch_size,).
                - pms: Phase margin specification indices (batch_size,).
                
        Returns:
            Dictionary with 'spec_dists':
                - If VAE mode: Tuple of (mu, logvar) for latent distribution.
                - Otherwise: Latent representation tensor of shape (batch_size, latent_dim).
        """
        gains, bws, pms = batch['gains'], batch['bws'], batch['pms']

        gain_embs = self.gain_embed(gains)
        bw_embs = self.bw_embed(bws)
        pm_embs = self.pm_embed(pms)

        conds = torch.cat([gain_embs, bw_embs, pm_embs], dim=-1)
        conds = self.cond_proj(conds)
        
        if self.vae:
            mu = self.fc_mu(conds)
            logvar = self.fc_logvar(conds)
            return {'spec_dists': (mu, logvar)}
        else:
            spec_latents = self.fc_latent(conds)
            return {'spec_dists': spec_latents}
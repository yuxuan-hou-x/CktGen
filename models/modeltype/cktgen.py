"""CktGen: Conditional Variational Autoencoder for specification-driven circuit generation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module implements CktGen, a conditional VAE that enables specification-driven
analog circuit generation through cross-modal alignment, contrastive learning, and
classifier-free guidance.
"""

import torch
import torch.nn as nn
from models.tools.losses import KLLoss, InfoNCE_with_filtering


class CKTGEN(nn.Module):
    """Conditional VAE for specification-driven circuit generation.
    
    CktGen extends VAE with conditioning on performance specifications using:
    1. Cross-modal alignment: Aligns circuit and spec embeddings in latent space.
    2. Contrastive learning: InfoNCE loss with false negative filtering.
    3. Classifier-free guidance: Predicts specifications from circuit latents.
    
    Attributes:
        archi: Encoder-decoder architecture for circuits.
        spec_encoder: Encoder for performance specifications.
        vae: If True, uses probabilistic latents; otherwise deterministic.
        conditioned: If True, conditions generation on specifications.
        contrastive: If True, applies InfoNCE contrastive loss.
        guided: If True, learns specification classifiers for guidance.
        filter: If True, filters false negatives in contrastive learning.
    """
    
    def __init__(self, archi, spec_encoder, lambdas, emb_dim, latent_dim, num_gain_type, num_pm_type, num_bw_type, eps_factor,
                 temperature, device, vae=False, conditioned=False, guided=False, contrastive=False, filter=False, **kwargs):
        """Initializes the CktGen model.
        
        Args:
            archi: Circuit encoder-decoder architecture.
            spec_encoder: Specification encoder module.
            lambdas: Dictionary of loss weights (e.g., {'kl': 1e-5, 'recon': 1.0, 'align': 1.0}).
            emb_dim: Node embedding dimension.
            latent_dim: Latent space dimension.
            num_gain_type: Number of discretized gain bins for classification.
            num_pm_type: Number of discretized phase margin bins.
            num_bw_type: Number of discretized bandwidth bins.
            eps_factor: Noise scaling for reparameterization trick.
            temperature: Temperature parameter for InfoNCE loss.
            device: Torch device (cuda/cpu).
            vae: If True, uses probabilistic encoding; if False, deterministic.
            conditioned: If True, enables specification conditioning.
            guided: If True, enables classifier-free guidance.
            contrastive: If True, applies InfoNCE contrastive loss.
            filter: If True, filters false negatives in contrastive learning.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.archi  = archi
        self.device = device 

        self.vae            = vae
        self.eps_factor     = eps_factor
        self.conditioned    = conditioned
        self.contrastive    = contrastive
        self.guided         = guided
        self.filter         = filter

        self.latent_dim     = latent_dim
        self.emb_dim        = emb_dim
        self.lambdas        = lambdas
        self.temperature    = temperature

        self.kl_loss_fn = KLLoss()
        
        # Specification encoder for conditioning
        self.spec_encoder = spec_encoder

        # Conditional losses and classifiers
        if self.conditioned:
            self.align_loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
            self.infonce_loss_fn = InfoNCE_with_filtering(temperature=self.temperature)
            self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')

            # Specification classifiers for guidance
            self.fc_gain = nn.Sequential(nn.Linear(latent_dim, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, num_gain_type))

            self.fc_pm   = nn.Sequential(nn.Linear(latent_dim, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, num_pm_type))
                                            
            self.fc_bw   = nn.Sequential(nn.Linear(latent_dim, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, num_bw_type))

        
    def forward(self, args, batch):
        """Forward pass for conditional generation from specifications.
        
        Args:
            args: Training/inference arguments.
            batch: Dictionary containing specifications and other data.
            
        Returns:
            Decoded circuit output from the architecture.
        """
        spec_latents = self.sample_spec_latents(batch, sample_mean=False, return_dists=False)
        batch['ckt_latents'] = spec_latents
        return self.archi.decode(args, batch)


    def sample_from_distribution(self, dist, sample_mean=False):
        """Samples from latent distribution using reparameterization trick.
        
        Args:
            dist: Tuple of (mu, logvar) representing Gaussian distribution.
            sample_mean: If True, returns mean; if False, samples stochastically.
            
        Returns:
            Sampled or deterministic latent vector.
        """
        mu, logvar = dist

        if sample_mean:
            return mu
        else:
            # Reparameterization: z = μ + σ * ε
            std = torch.exp(logvar / 2)
            eps = std.data.new(std.size()).normal_() * self.eps_factor
            return eps.mul(std).add_(mu)


    def sample_ckt_latents(self, batch, sample_mean=False, return_dists=False):
        """Encodes circuits into latent space.
        
        Args:
            batch: Dictionary containing circuit data.
            sample_mean: If True, uses mean of distribution (deterministic).
            return_dists: If True, also returns distribution parameters.
            
        Returns:
            Circuit latent vectors, optionally with distribution parameters.
        """
        batch.update(self.archi.encode(batch))

        if self.vae:
            ckt_latents = self.sample_from_distribution(batch['ckt_dists'], sample_mean=sample_mean).squeeze()
        else:
            ckt_latents = batch['ckt_dists']

        if return_dists and self.vae:
            return (ckt_latents.squeeze(), batch['ckt_dists'])
        else:
            return ckt_latents.squeeze()
                

    def sample_spec_latents(self, batch, sample_mean=False, return_dists=False):
        """Encodes specifications into latent space.
        
        Args:
            batch: Dictionary containing specification data.
            sample_mean: If True, uses mean of distribution (deterministic).
            return_dists: If True, also returns distribution parameters.
            
        Returns:
            Specification latent vectors, optionally with distribution parameters.
        """
        batch.update(self.spec_encoder(batch))
        if self.vae:
            spec_latents = self.sample_from_distribution(batch['spec_dists'], sample_mean=sample_mean).squeeze()
        else:
            spec_latents = batch['spec_dists']

        if return_dists and self.vae:
            return (spec_latents.squeeze(), batch['spec_dists'])
        else:
            return spec_latents.squeeze()


    def compute_loss(self, batch):
        """Computes multi-task loss for CktGen.
        
        Combines multiple losses:
        1. Reconstruction loss: Circuit reconstruction quality.
        2. KL divergence: Regularizes latent distributions.
        3. Alignment loss: Aligns circuit and spec embeddings.
        4. Contrastive loss: InfoNCE for discrimination.
        5. Guidance loss: Specification classification.
        
        Args:
            batch: Dictionary containing circuits, specifications, and metadata.
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        losses = self.archi.compute_loss(batch)

        if self.vae:
            ckt_latents, ckt_dists = batch['ckt_latents'], batch['ckt_dists']
            spec_latents, spec_dists = self.sample_spec_latents(batch, sample_mean=False, return_dists=True)
        else:
            ckt_latents = batch['ckt_latents']
            spec_latents = self.sample_spec_latents(batch, sample_mean=False, return_dists=False)
        
        if self.conditioned:
            batch['ckt_latents'] = spec_latents
            losses_spec2ckt = self.archi.compute_loss(batch)
            losses['recon'] = losses['recon'] + losses_spec2ckt['recon']

            losses['align'] = self.align_loss_fn(spec_latents, ckt_latents)

            if self.contrastive:
                if self.filter:
                    losses['nce'] = self.infonce_loss_fn(spec_latents, ckt_latents, mask=batch['filter_mask']) # sample by reparam
                else:
                    losses['nce'] = self.infonce_loss_fn(spec_latents, ckt_latents, mask=None) # sample by reparam

            if self.guided:
                gnd_gains, gnd_bws, gnd_pms = batch['gains'], batch['bws'], batch['pms']
                logits_gain = self.fc_gain(ckt_latents)
                logits_bw   = self.fc_bw(ckt_latents)
                logits_pm   = self.fc_pm(ckt_latents)

                loss_gain     = self.ce_loss_fn(logits_gain, gnd_gains)
                loss_bw       = self.ce_loss_fn(logits_bw, gnd_bws)
                loss_pm       = self.ce_loss_fn(logits_pm, gnd_pms)

                losses['gde'] = loss_gain + loss_bw + loss_pm

        # if self.vae:
        #     mu, logvar = batch['ckt_dists']
        #     losses['kl'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.vae:
            ref_mus = torch.zeros_like(ckt_dists[0])
            ref_logvar = torch.zeros_like(ckt_dists[1]) # logvar = 0 -> std = 1
            ref_dists = (ref_mus, ref_logvar)         # normal gaussian distribution
            
            losses["kl"] = (
                self.kl_loss_fn(spec_dists, ckt_dists)          # spec to ckt
                + self.kl_loss_fn(ckt_dists, spec_dists)        # ckt to spec
                + self.kl_loss_fn(ckt_dists, ref_dists)         # ckt to normal gaussian
                + self.kl_loss_fn(spec_dists, ref_dists)        # spec to normal gaussian
            )
        
        mixed_loss = 0
        for ltype, lam in self.lambdas.items():
            mixed_loss += losses[ltype] * lam

        return mixed_loss, losses


    def predict_specificaetion(self, ckt_latents, topk=1):
        """Predicts specifications from circuit latents (for guidance).
        
        Args:
            ckt_latents: Circuit latent vectors of shape [batch_size, latent_dim].
            topk: Number of top predictions to return.
            
        Returns:
            Dictionary with keys 'gain', 'bw', 'pm', each containing top-k predictions.
        """
        logits_gain = self.fc_gain(ckt_latents)
        logits_bw   = self.fc_bw(ckt_latents)
        logits_pm   = self.fc_pm(ckt_latents)

        _, pred_gain = torch.topk(logits_gain, k=topk, dim=-1)
        _, pred_bw   = torch.topk(logits_bw, k=topk, dim=-1)
        _, pred_pm   = torch.topk(logits_pm, k=topk, dim=-1)

        return {'gain': pred_gain, 'bw': pred_bw, 'pm': pred_pm}
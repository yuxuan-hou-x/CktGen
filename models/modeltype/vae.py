"""Variational Autoencoder (VAE) for circuit generation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module implements a Variational Autoencoder (VAE) for learning 
latent representations of analog circuit topologies and enabling 
random circuit generation through sampling from the learned prior.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for circuit topology learning.
    
    This class implements a VAE that learns to encode circuit graphs into
    a continuous latent space and reconstruct them. The model uses the
    reparameterization trick for gradient-based learning and combines
    reconstruction loss with KL divergence regularization.
    
    Attributes:
        archi: The encoder-decoder architecture (e.g., CktArchi, PACE).
        latent_dim: Dimensionality of the latent space.
        emb_dim: Dimensionality of node embeddings.
        lambdas: Dictionary mapping loss types to their weights.
        eps_factor: Scaling factor for sampling noise in reparameterization.
    """
    
    def __init__(self, archi, lambdas, eps_factor, emb_dim, latent_dim, **kwargs):
        """Initializes the VAE model.
        
        Args:
            archi: The encoder-decoder architecture module.
            lambdas: Dictionary of loss weights, e.g., {'recon': 1.0, 'kl': 5e-3}.
            eps_factor: Noise scaling factor for reparameterization trick.
            emb_dim: Embedding dimension for nodes.
            latent_dim: Dimension of latent space.
            **kwargs: Additional arguments (for compatibility).
        """
        super().__init__()
        
        self.archi = archi
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.lambdas = lambdas
        self.eps_factor = eps_factor

    def sample_from_distribution(self, dist, sample_mean=False):
        """Samples from the latent distribution using reparameterization trick.
        
        Implements the reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1).
        This allows gradients to flow through the sampling operation during training.
        
        Args:
            dist: Tuple of (mu, logvar) representing the Gaussian distribution.
            sample_mean: If True, returns the mean (deterministic). If False,
                samples from the distribution (stochastic).
                
        Returns:
            Sampled latent vector(s) of shape [batch_size, latent_dim].
        """
        mu, logvar = dist
        if sample_mean:
            return mu
        else:
            # Reparameterization trick: z = μ + σ * ε
            std = torch.exp(logvar / 2)
            eps = std.data.new(std.size()).normal_() * self.eps_factor
            return eps.mul(std).add_(mu)

    def compute_loss(self, batch):
        """Computes the VAE loss (ELBO = reconstruction + KL divergence).
        
        The Evidence Lower Bound (ELBO) objective consists of:
        1. Reconstruction loss: measures how well the decoder reconstructs input.
        2. KL divergence: regularizes latent space to be close to N(0, I).
        
        Args:
            batch: Dictionary containing:
                - 'ckt_dists': Tuple of (mu, logvar) from encoder.
                - Other keys required by architecture's compute_loss.
                
        Returns:
            Tuple of (mixed_loss, losses_dict) where:
                - mixed_loss: Weighted sum of all losses (scalar).
                - losses_dict: Dictionary with individual loss values.
        """
        # Compute reconstruction loss from architecture
        losses = self.archi.compute_loss(batch)
        
        # Compute KL divergence: KL(q(z|x) || p(z))
        # where q(z|x) = N(μ, σ²) and p(z) = N(0, I)
        mu, logvar = batch['ckt_dists']
        losses['kl'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine losses with their weights
        mixed_loss = 0
        for ltype, lam in self.lambdas.items():
            mixed_loss += losses[ltype] * lam

        return mixed_loss, losses
"""Denoiser module for diffusion models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
from torch import nn
from models.modeltype.ldt_module.pos_encoding import build_position_encoding
from models.modeltype.ldt_module.timestep_embedding import TimestepEmbedding, Timesteps
from models.modeltype.ldt_module.transformer_diffusion import (
    TransformerEncoderLayer,
    SkipTransformerEncoder,
)

class Denoiser(nn.Module):
    """Denoiser network for diffusion-based latent generation.
    
    This module denoises latent representations conditioned on timestep and
    specification embeddings using a transformer encoder architecture.
    
    Args:
        latent_dim: Dimension of latent representations.
        ff_size: Feedforward layer size in transformer.
        num_layers: Number of transformer encoder layers (adjusted to be odd).
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        normalize_before: If True, apply layer norm before attention/FFN.
        activation: Activation function for transformer ('gelu', 'relu', etc.).
        flip_sin_to_cos: If True, flip sine and cosine in timestep embeddings.
        freq_shift: Frequency shift for timestep embeddings.
        condition_dim: Dimension of conditioning embeddings.
        **kwargs: Additional arguments.
    """
    def __init__(
        self,
        latent_dim: int = 64,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        normalize_before: bool = False,
        activation: str = 'gelu',
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        condition_dim: int = 64,
        **kwargs
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.time_proj = Timesteps(condition_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(condition_dim, self.latent_dim)
        if condition_dim != self.latent_dim:
            self.condition_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(condition_dim, self.latent_dim)
            )
        num_layers = num_layers if num_layers % 2 == 1 else num_layers + 1

        self.query_pos = build_position_encoding(self.latent_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm
        )

    def forward(self, sample, timestep, condition):
        """Denoise latent samples conditioned on timestep and specifications.
        
        Args:
            sample: Noisy latent samples of shape (batch_size, 1, latent_dim).
            timestep: Diffusion timestep (scalar or batch).
            condition: Conditioning embeddings of shape (batch_size, 1, condition_dim).
            
        Returns:
            Denoised latent samples of shape (batch_size, 1, latent_dim).
        """
        sample = sample.permute(1, 0, 2)        # (1, bsz, latent_dim)
        condition = condition.permute(1, 0, 2)

        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        if self.condition_dim != self.latent_dim:
            condition = self.condition_proj(condition)

        # print(time_emb.size(), condition.size())
        emb_latent = torch.cat((time_emb, condition), 0)

        xseq = torch.cat((sample, emb_latent), axis=0)

        xseq = self.query_pos(xseq)         # (3, bsz, latent_dim) -> (latent, time_emb, condition)

        tokens = self.encoder(xseq)         # (3, bsz, latent_dim)
        sample = tokens[: sample.shape[0]]  # (1, bsz, latent_dim)

        sample = sample.permute(1, 0, 2) # (bsz, 1, latent_dim)

        return sample
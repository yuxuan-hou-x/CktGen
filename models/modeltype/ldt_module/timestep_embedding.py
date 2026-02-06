"""Timestep embedding for diffusion models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
import math
from torch import nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """Create sinusoidal timestep embeddings for diffusion models.
    
    This matches the implementation in Denoising Diffusion Probabilistic Models.
    
    Args:
        timesteps: 1-D tensor of N timestep indices, one per batch element (may be fractional).
        embedding_dim: Dimension of the output embeddings.
        flip_sin_to_cos: If True, flip sine and cosine components.
        downscale_freq_shift: Shift applied to frequency calculation.
        scale: Scaling factor for embeddings.
        max_period: Controls the minimum frequency of the embeddings.
        
    Returns:
        Tensor of shape [N x embedding_dim] containing positional embeddings.
    """
    
    assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Module for converting timestep indices to sinusoidal embeddings.
    
    Args:
        num_channels: Number of embedding channels.
        flip_sin_to_cos: If True, flip sine and cosine components.
        downscale_freq_shift: Frequency shift parameter.
    """
    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        """Convert timestep indices to embeddings.
        
        Args:
            timesteps: 1-D tensor of timestep indices.
            
        Returns:
            Timestep embeddings tensor.
        """
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    """MLP for processing timestep embeddings.
    
    Projects sinusoidal timestep embeddings through a two-layer MLP with
    optional activation function.
    
    Args:
        channel: Input channel dimension.
        time_embed_dim: Output embedding dimension.
        act_fn: Activation function ('silu' or None).
    """
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = 'silu'):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == 'silu':
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        """Process timestep embeddings through MLP.
        
        Args:
            sample: Input timestep embeddings.
            
        Returns:
            Processed timestep embeddings.
        """
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample

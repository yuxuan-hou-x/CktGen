"""Various positional encodings for the transformer.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""
import math
import torch

import numpy as np
from torch import nn, Tensor
from typing import List, Optional

 
def build_position_encoding(dim):
    """Build a position embedding module.
    
    Args:
        dim: Embedding dimension.
        
    Returns:
        PositionEmbedding module.
    """
    position_embedding = PositionEmbedding(dim)
    return position_embedding


def PE1d_sincos(dim, seq_length=3):
    """Generate 1D sinusoidal positional encodings.
    
    Args:
        dim: Dimension of the model (must be even).
        seq_length: Length of the sequence.
        
    Returns:
        Positional encoding tensor of shape (seq_length, 1, dim).
    """
    if dim % 2 != 0:
        raise ValueError('Cannot use sin/cos positional encoding with '
                         'odd dim (got dim={:d})'.format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * 
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """Absolute positional embedding module.
    
    Uses sinusoidal positional encodings that can optionally be learned.
    
    Args:
        dim: Feature dimension.
        seq_length: Sequence length.
        dropout: Dropout probability.
        grad: Whether to make embeddings learnable (default: False).
    """

    def __init__(self, dim, seq_length=3, dropout=0.1, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(dim, seq_length), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Add positional embeddings to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, feat_dim).
            
        Returns:
            Input with positional embeddings added, same shape as input.
        """
        # x.shape: seq_len, bs, feat_dim
        pos_x = x + self.embed.expand(x.shape)
        pos_x = self.dropout(pos_x)
        return pos_x
"""Transformer-based diffusion architecture.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Optional
import copy


def _get_activation_fn(activation):
    """Return an activation function given a string.
    
    Args:
        activation: Name of activation function ('relu', 'gelu', 'glu').
        
    Returns:
        Activation function.
    """
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def _get_clones(module, N):
    """Create N deep copies of a module.
    
    Args:
        module: Module to clone.
        N: Number of copies.
        
    Returns:
        ModuleList containing N clones.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_clone(module):
    """Create a deep copy of a module.
    
    Args:
        module: Module to clone.
        
    Returns:
        Deep copy of the module.
    """
    return copy.deepcopy(module)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feedforward network.
    
    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout probability.
        activation: Activation function name.
        normalize_before: If True, apply layer norm before attention/FFN (pre-norm).
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embeddings to tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass with post-normalization (norm after attention/FFN)."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, 
            k, 
            value=src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass with pre-normalization (norm before attention/FFN)."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, 
            k, 
            value=src2, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class SkipTransformerEncoder(nn.Module):
    """Transformer encoder with skip connections (U-Net style).
    
    Uses a U-Net-like architecture with input blocks, middle block, and output
    blocks with skip connections between corresponding input and output blocks.
    
    Args:
        encoder_layer: Encoder layer template to clone.
        num_layers: Total number of layers (must be odd).
        norm: Optional normalization layer.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass with skip connections.
        
        Args:
            src: Input tensor.
            mask: Attention mask.
            src_key_padding_mask: Padding mask for keys.
            pos: Positional embeddings.
            
        Returns:
            Encoded tensor with skip connections applied.
        """
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(
                x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
            xs.append(x)

        x = self.middle_block(
            x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipTransformerDecoder(nn.Module):
    """Transformer decoder with skip connections (U-Net style).
    
    Similar to SkipTransformerEncoder but for decoder architecture with
    cross-attention to encoder memory.
    
    Args:
        decoder_layer: Decoder layer template to clone.
        num_layers: Total number of layers (must be odd).
        norm: Optional normalization layer.
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass with skip connections and cross-attention.
        
        Args:
            tgt: Target sequence.
            memory: Encoder output to attend to.
            tgt_mask: Target attention mask.
            memory_mask: Memory attention mask.
            tgt_key_padding_mask: Target padding mask.
            memory_key_padding_mask: Memory padding mask.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for target.
            
        Returns:
            Decoded tensor with skip connections applied.
        """
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            xs.append(x)

        x = self.middle_block(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            x = self.norm(x)

        return x

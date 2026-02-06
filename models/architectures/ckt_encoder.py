"""Circuit encoder module for graph-based representations.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import math
import torch
from torch import nn
from models.architectures.tools.gin import GIN



def PE1d_sincos(seq_length, dim):
    """Generates 1D sinusoidal positional encodings.
    
    Creates positional encodings using sine and cosine functions at different
    frequencies, as described in "Attention is All You Need".
    
    Args:
        seq_length: Length of the sequence (number of positions).
        dim: Dimension of the positional encoding (must be even).
        
    Returns:
        Tensor of shape (seq_length, 1, dim) containing positional encodings.
        
    Raises:
        ValueError: If dim is odd.
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
    
    Provides sinusoidal positional embeddings that can optionally be learned
    during training. Used to inject position information into the model.
    
    Args:
        seq_length: Maximum sequence length.
        dim: Embedding dimension.
        dropout: Dropout probability.
        grad: Whether to allow gradients (learnable embeddings).
        is_decoder: Whether this is used in a decoder (affects broadcasting).
    """

    def __init__(self, seq_length, dim, dropout, grad=False, is_decoder=True):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        self.is_decoder = is_decoder

    def forward(self, x):
        """Adds positional embeddings to input.
        
        Args:
            x: Input tensor. Shape depends on is_decoder flag:
                - Decoder: (1, batch_size, feat_dim)
                - Encoder: (seq_len, batch_size, feat_dim)
                
        Returns:
            Tensor with positional embeddings added, same shape as input.
        """
        if self.is_decoder:
            # x.shape: 1, bs, feat_dim
            pos_x = self.embed.repeat(1, x.shape[1], 1)
        else:
            # x.shape: seq_len, bs, feat_dim
            pos_x = x + self.embed.expand(x.shape)
        pos_x = self.dropout(pos_x)
        return pos_x


class Ckt_TransEncoder(nn.Module):
    """Transformer encoder for circuit graphs.
    
    Encodes circuit graphs into latent representations using transformer architecture.
    Combines vertex type, path, topology, and size information through embeddings
    and multi-head self-attention.
    
    Args:
        max_n: Maximum number of vertices in a circuit.
        emb_dim: Embedding dimension for graph features.
        hidden_dim: Hidden dimension for transformer.
        latent_dim: Dimension of output latent representation.
        ff_size: Feedforward layer size in transformer.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        num_types: Number of component types.
        num_paths: Number of signal paths.
        size_emb_dim: Embedding dimension for component sizes.
        activation: Activation function ('gelu' or 'relu').
        dropout: Dictionary with dropout rates for 'pos', 'trans', and 'graph'.
        pos_grad: Whether positional embeddings are learnable.
        device: Device to run the model on.
        vae: Whether to use VAE mode (outputs mu and logvar).
        **kwargs: Additional arguments.
    """
    def __init__(
        self, max_n=8, emb_dim=128, hidden_dim=512, latent_dim=64, ff_size=512, num_layers=4, num_heads=8, num_types=26, 
        num_paths=8, size_emb_dim=8, activation='gelu', dropout={'pos': 0.2, 'trans': 0.1, 'graph': 0.2}, 
        pos_grad=False, device=None, vae=False, **kwargs
    ):
        
        super().__init__()
        self.max_n = max_n
        self.vae = vae

        if self.vae:
            self.token_num  = 2
            self.mu_token   = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.std_token  = nn.Parameter(torch.randn(1, 1, hidden_dim))
        else:
            self.token_num  = 1
            self.latent_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.type_embed = nn.Embedding(num_types, emb_dim)
        self.path_embed = nn.Embedding(num_paths, emb_dim)
        self.topo_embed = GIN(emb_dim, dropout['graph'], max_n)

        self.size_embed = nn.Sequential(nn.Linear(self.max_n * 3, emb_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim, size_emb_dim))

        self.posi_embed = PositionEmbedding(max_n + self.token_num, 
                                            hidden_dim, 
                                            dropout['pos'], 
                                            pos_grad, 
                                            False)

        self.graph_mlp = nn.Linear(emb_dim * 3, hidden_dim)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout['trans'],
                                                          activation=activation)
        
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, 
                                                     num_layers=num_layers)

        if self.vae:
            self.fc_mu      = nn.Linear(hidden_dim+size_emb_dim, latent_dim)
            self.fc_logvar  = nn.Linear(hidden_dim+size_emb_dim, latent_dim)
        else:
            self.fc_latent = nn.Linear(hidden_dim+size_emb_dim, latent_dim)

    def forward(self, batch):
        """Encodes circuit graphs into latent representations.
        
        Args:
            batch: Dictionary containing:
                - v_types: Vertex types (batch_size, max_n).
                - v_paths: Vertex paths (batch_size, max_n).
                - v_topos_1hot: One-hot topology features.
                - v_sizes: Vertex sizes (batch_size, max_n * 3).
                - adj: Adjacency matrices.
                - spec_embs (optional): Specification embeddings to add.
                
        Returns:
            Dictionary with 'ckt_dists':
                - If VAE mode: Tuple of (mu, logvar) for latent distribution.
                - Otherwise: Latent representation tensor.
        """
        v_types, v_paths, v_topos_1hot, v_sizes, adj = (batch['v_types'], 
                                                        batch['v_paths'], 
                                                        batch['v_topos_1hot'], 
                                                        batch['v_sizes'], 
                                                        batch['adj'])
        bs, _ = v_types.shape

        v_type_embs = self.type_embed(v_types)  # (bs, max_n, emb_dim)
        v_path_embs = self.path_embed(v_paths)  # (bs, max_n, emb_dim)
        v_topo_embs = self.topo_embed(v_topos_1hot, adj) # (bsz, max_n, max_n) -> (bsz, max_n, node_emb_dim)
        
        g_cat = torch.cat((v_type_embs.transpose(0, 1),
                           v_path_embs.transpose(0, 1),
                           v_topo_embs.transpose(0, 1),), dim=-1)
                           
        x = self.graph_mlp(g_cat)

        # add specification
        if 'spec_embs' in batch:
            x = x + batch['spec_embs']


        if self.vae:
            mu_token = self.mu_token.repeat(1, bs, 1)  # (1, bs, latent_dim)
            std_token = self.std_token.repeat(1, bs, 1)  # (1, bs, latent_dim)
            xseq = torch.cat((mu_token, std_token, x), dim=0)
        else:
            latent_token = self.latent_token.repeat(1, bs, 1)
            xseq = torch.cat((latent_token, x), dim=0)

        xseq = self.posi_embed(xseq)
        xseq = self.seqTransEncoder(xseq)
        size_embs = self.size_embed(v_sizes)


        if self.vae:
            mu, logvar = xseq[0], xseq[1]
            mu, logvar = torch.cat([mu, size_embs], dim=-1), torch.cat([logvar, size_embs], dim=-1)
            mu, logvar = self.fc_mu(mu).unsqueeze(0), self.fc_logvar(logvar).unsqueeze(0)
            return {'ckt_dists': (mu, logvar)}
        else:
            latent = torch.cat([xseq[0], size_embs], dim=-1)
            latent = self.fc_latent(latent).unsqueeze(0)
            return {'ckt_dists': latent}
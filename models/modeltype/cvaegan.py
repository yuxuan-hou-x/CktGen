"""CVAEGAN: Conditional VAE-GAN for adversarial circuit generation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
import torch.nn as nn

from models.architectures.tools.gin import GIN
from models.architectures.ckt_encoder import PositionEmbedding


class Spec_Embedder(nn.Module):
    """Specification embedder for CVAEGAN.
    
    Encodes performance specifications for conditioning the VAE-GAN.
    
    Args:
        num_gain_type: Number of discretized gain levels.
        num_pm_type: Number of discretized phase margin levels.
        num_bw_type: Number of discretized bandwidth levels.
        emb_dim: Embedding dimension.
        latent_dim: Output latent dimension.
    """
    def __init__(self, num_gain_type, num_pm_type, num_bw_type, emb_dim=128, latent_dim=64):
        super().__init__()

        self.gain_embed = nn.Embedding(num_gain_type, emb_dim)
        self.pm_embed   = nn.Embedding(num_pm_type, emb_dim)
        self.bw_embed   = nn.Embedding(num_bw_type, emb_dim)

        self.cond_proj = nn.Linear(emb_dim*3, emb_dim)
        self.fc_embs   = nn.Linear(emb_dim, latent_dim)

    def forward(self, gains, bws, pms):
        """Encode specifications to embeddings.
        
        Args:
            gains: Gain specification indices.
            bws: Bandwidth specification indices.
            pms: Phase margin specification indices.
            
        Returns:
            Spec embeddings of shape (batch_size, latent_dim).
        """
        gain_embs = self.gain_embed(gains)
        bw_embs = self.bw_embed(bws)
        pm_embs = self.pm_embed(pms)

        conditions = torch.cat([gain_embs, bw_embs, pm_embs], dim=-1)
        conditions = self.cond_proj(conditions)

        spec_embs = self.fc_embs(conditions)
        return spec_embs



class Discriminator(nn.Module):
    """Discriminator network for CVAEGAN.
    
    Discriminates between real and generated circuits based on circuit structure.
    
    Args:
        device: Device to run the model on.
        num_gain_type: Number of gain levels.
        num_pm_type: Number of phase margin levels.
        num_bw_type: Number of bandwidth levels.
        max_n: Maximum number of vertices.
        emb_dim: Embedding dimension.
        hidden_dim: Hidden dimension for transformer.
        latent_dim: Latent dimension.
        ff_size: Feedforward size in transformer.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_types: Number of component types.
        num_paths: Number of signal paths.
        size_emb_dim: Embedding dimension for component sizes.
        activation: Activation function.
        dropout: Dropout rates dictionary.
        pos_grad: Whether position embeddings are learnable.
        **kwargs: Additional arguments.
    """
    def __init__(
        self, device, num_gain_type, num_pm_type, num_bw_type, max_n=8, emb_dim=128, hidden_dim=512, latent_dim=64, ff_size=512,
        num_layers=4, num_heads=8, num_types=26, num_paths=8, size_emb_dim=8, activation='gelu', 
        dropout={'pos': 0.2, 'trans': 0.1, 'graph': 0.2}, pos_grad=False, **kwargs
    ):
        
        super().__init__()
        self.type = type
        self.max_n = max_n
        self.latent_dim=latent_dim

        self.latent_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.type_embed = nn.Embedding(num_types, emb_dim)
        self.path_embed = nn.Embedding(num_paths, emb_dim)
        self.topo_embed = GIN(emb_dim, dropout['graph'], max_n)

        self.size_embed = nn.Sequential(nn.Linear(self.max_n * 3, emb_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim, size_emb_dim))

        self.posi_embed = PositionEmbedding(max_n + 1, 
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

        self.fc_latent = nn.Linear(hidden_dim+size_emb_dim, latent_dim)
    
        self.prob_fc = nn.Sequential(nn.Linear(latent_dim, latent_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(latent_dim, 1))  # whether to add edge between v_i and v_new, f(hvi, hnew


    def forward(self, batch):
        """Discriminate circuit validity.
        
        Args:
            batch: Batch containing circuit data.
            
        Returns:
            Logits indicating whether circuits are real (shape: batch_size, 1).
        """
        v_types, v_paths, v_topos_1hot, v_sizes, adj = (batch['v_types'], 
                                                        batch['v_paths'], 
                                                        batch['v_topos_1hot'], 
                                                        batch['v_sizes'], 
                                                        batch['adj'])

        gains, bws, pms = batch['gains'], batch['bws'], batch['pms']
        bs, _ = v_types.shape

        v_type_embs = self.type_embed(v_types)  # (bs, max_n, emb_dim)
        v_path_embs = self.path_embed(v_paths)  # (bs, max_n, emb_dim)
        v_topo_embs = self.topo_embed(v_topos_1hot, adj) # (bsz, max_n, max_n) -> (bsz, max_n, node_emb_dim)
        
        g_cat = torch.cat((v_type_embs.transpose(0, 1),
                           v_path_embs.transpose(0, 1),
                           v_topo_embs.transpose(0, 1),), dim=-1)
                           
        x = self.graph_mlp(g_cat)
        latent_token = self.latent_token.repeat(1, bs, 1)
        xseq = torch.cat((latent_token, x), dim=0)
        xseq = self.posi_embed(xseq)
        xseq = self.seqTransEncoder(xseq)
        size_embs = self.size_embed(v_sizes)

        latents = torch.cat([xseq[0], size_embs], dim=-1)
        latents = self.fc_latent(latents)
        
        return self.prob_fc(latents)
        

class CVAEGAN(nn.Module):
    """Conditional Variational Autoencoder GAN for circuit generation.
    
    Combines VAE with adversarial training to generate high-quality circuits
    conditioned on performance specifications.
    
    Args:
        archi: Base architecture (e.g., CktGen) providing encode/decode.
        discriminator: Discriminator network.
        lambdas: Loss weight dictionary (e.g., {'kl': 0.1, 'recon': 1.0}).
        num_gain_type: Number of gain levels.
        num_pm_type: Number of phase margin levels.
        num_bw_type: Number of bandwidth levels.
        emb_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        latent_dim: Latent dimension.
        device: Device to run the model on.
        **kwargs: Additional arguments.
    """
    def __init__(
        self, archi, discriminator, lambdas, num_gain_type, num_pm_type, num_bw_type,
        emb_dim=128, hidden_dim=512, latent_dim=64, device=None, **kwargs
    ):
        super().__init__()
        self.archi = archi
        self.discriminator = discriminator
        self.spec_embedder = Spec_Embedder(
            num_gain_type=num_gain_type, 
            num_pm_type=num_pm_type, 
            num_bw_type=num_bw_type, 
            emb_dim=emb_dim, 
            latent_dim=latent_dim
        )
        self.eps_factor = 1
        self.lambdas = lambdas

        self.adver_loss_fn = torch.nn.MSELoss()
        self.spec_project = nn.Linear(latent_dim, hidden_dim)


    def forward(self, args, batch):
        """Generate circuits from latents conditioned on specifications.
        
        Args:
            args: Additional arguments for decoding.
            batch: Batch containing circuit latents and specifications.
            
        Returns:
            Decoded circuit outputs.
        """
        spec_embs = self.spec_embedder(batch['gains'], batch['bws'], batch['pms'])
        ckt_latents = batch['ckt_latents']

        batch['ckt_latents'] = ckt_latents + spec_embs
        return self.archi.decode(args, batch)


    def sample_from_distribution(self, dist, sample_mean=False):
        """Sample from a Gaussian distribution.
        
        Args:
            dist: Tuple of (mu, logvar).
            sample_mean: If True, return mean; otherwise sample.
            
        Returns:
            Sampled latent vector.
        """
        mu, logvar = dist

        if sample_mean:
            return mu
        else:
            std = torch.exp(logvar / 2)
            eps = std.data.new(std.size()).normal_() * self.eps_factor
            return eps.mul(std).add_(mu)


    def sample_ckt_latents(self, batch, sample_mean=False, return_dists=False):
        """Sample circuit latents from encoder distribution.
        
        Args:
            batch: Input batch.
            sample_mean: If True, return mean of distribution.
            return_dists: If True, also return distribution parameters.
            
        Returns:
            Sampled circuit latents (and optionally distribution parameters).
        """
        batch.update(self.archi.encode(batch))
        ckt_latents = self.sample_from_distribution(batch['ckt_dists'], sample_mean=sample_mean).squeeze()
        if return_dists:
            return (ckt_latents.squeeze(), batch['ckt_dists'])
        else:
            return ckt_latents.squeeze()


    def compute_cvae_loss(self, batch):
        """Compute conditional VAE loss.
        
        Args:
            batch: Training batch.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        mu, logvar = batch['ckt_dists']
        ckt_latents = batch['ckt_latents']
        spec_embs = self.spec_embedder(batch['gains'], batch['bws'], batch['pms'])

        batch['ckt_latents'] = ckt_latents + spec_embs
        losses = self.archi.compute_loss(batch)
        losses['kl'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        mixed_loss = 0
        for ltype, lam in self.lambdas.items():
            mixed_loss += losses[ltype] * lam

        return mixed_loss, losses


    def compute_discriminator_loss(self, batch, label):
        """Compute discriminator loss.
        
        Args:
            batch: Batch containing circuit data.
            label: Target labels (1 for real, 0 for fake).
            
        Returns:
            Discriminator loss.
        """
        logits_valid = self.discriminator(batch)
        loss =  self.adver_loss_fn(logits_valid, label)
        return loss


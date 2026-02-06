"""Circuit decoder module for autoregressive circuit generation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import math
import torch
import torch.nn as nn
import numpy as np
import utils.eval_reconstruct as utils_eval_reconstruct
import utils.data as utils_data
from torch.nn import functional as F
from models.architectures.tools.gin import GIN


class Ckt_TransDecoder(nn.Module):
    """Transformer-based autoregressive decoder for circuit generation.
    
    This decoder generates circuits autoregressively, predicting one vertex at a time
    along with its connections to previous vertices. It uses cross-attention between
    the circuit latent representation and the partially generated graph.
    
    Args:
        device: Device to run the model on (CPU/GPU).
        max_n: Maximum number of vertices in a circuit.
        num_types: Number of different component types.
        num_paths: Number of different signal paths in the circuit.
        hidden_dim: Hidden state dimension.
        latent_dim: Dimension of latent circuit representation.
        emb_dim: Embedding dimension for vertices.
        size_emb_dim: Embedding dimension for component sizes.
        block_size: Maximum sequence length for transformer blocks.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        fc_rate: Expansion factor for feedforward layers.
        type_rate: Loss weight for component type prediction.
        path_rate: Loss weight for path prediction.
        size_rate: Loss weight for component size prediction.
        **kwargs: Additional arguments.
    """
    def __init__(self, device=None, max_n=8, num_types=26, num_paths=8, hidden_dim=512, latent_dim=64, emb_dim=128, size_emb_dim=8, block_size=9, 
                 num_layers=4, num_heads=8, dropout_rate=0.1, fc_rate=4, type_rate=0.3, path_rate=0.01, size_rate=0.01, **kwargs):
        
        super().__init__()
        
        self.device = device
        self.max_n = max_n 
        self.num_types = num_types
        self.num_paths = num_paths
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.type_rate=type_rate
        self.path_rate=path_rate
        self.size_rate=size_rate
        self.trans_cross = TransGPT_Cross(max_n, num_types, num_paths, hidden_dim, latent_dim, emb_dim, block_size, 
                                          num_layers, num_heads, dropout_rate, fc_rate)

        self.block_size = block_size
        self.size_decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                          nn.Softplus(),
                                          nn.Linear(hidden_dim, latent_dim),
                                          nn.Softplus(),
                                          nn.Linear(latent_dim, self.max_n * 3))

        self.edge_fc = nn.Sequential(nn.Linear(emb_dim*2, emb_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(emb_dim, 1))  # whether to add edge between v_i and v_new, f(hvi, hnew


    def get_block_size(self):
        """Returns the maximum sequence length for this decoder."""
        return self.block_size


    def forward(self, args, batch):
        """Generates circuits autoregressively from latent representations.
        
        Args:
            args: Configuration dictionary containing circuit parameters.
            batch: Dictionary containing 'ckt_latents' - latent circuit representations.
            
        Returns:
            List of generated circuit graphs (igraph objects).
        """
        ckt_latents = batch['ckt_latents']
        if len(ckt_latents.shape) == 2:
            ckt_latents = ckt_latents.unsqueeze(0)

        _, bsz, _ = ckt_latents.size()
        rec_sizes = self.size_decoder(ckt_latents)
        if len(rec_sizes.shape) > 2:
            rec_sizes = rec_sizes.squeeze(0)
        
        rec_graphs = utils_eval_reconstruct.init_recon_graph(args, bsz)
        finished = [False] * bsz
        
        for v in range(2, self.max_n + 1):
            _rec_graphs = utils_data.collate_fn(rec_graphs)
            batch = utils_data.transforms(args, _rec_graphs, mode='generate')
            batch['ckt_latents'] = ckt_latents
            
            outputs = self.trans_cross(batch)
            # print(outputs.shape)
            
            logits_type = outputs['logits_type'][:, v-1, :]
            logits_path = outputs['logits_path'][:, v-1, :]
            logits_topo = outputs['logits_topo'][:, v-1, :]

            #### --- use the max prob one --- ####
            probs_type = torch.softmax(logits_type, dim=-1)
            probs_path = torch.softmax(logits_path, dim=-1)
            _, new_types = torch.max(probs_type, dim=-1) # (bsz, 1)
            _, new_paths = torch.max(probs_path, dim=-1) # (bsz, 1)

            logits_edge = torch.cat([torch.stack([logits_topo]*(v-1), 1), 
                                     outputs['logits_topo'][:, :v-1, :]], -1)  # just from the center node to the target node
            probs_edge = torch.sigmoid(self.edge_fc(logits_edge))

            # reconstruct edges
            for bi, g in enumerate(rec_graphs):
                if finished[bi]:
                    continue
                else:
                    if v < self.max_n:
                        g.add_vertex(type=new_types[bi])
                    else:
                        g.add_vertex(type=args['END_TYPE'])
                
                    g.vs[v-1]['path']   = new_paths[bi]
                    g.vs[v-1]['r']      = rec_sizes[bi, new_paths[bi] * 3 + 0]
                    g.vs[v-1]['c']      = rec_sizes[bi, new_paths[bi] * 3 + 1]
                    g.vs[v-1]['gm']     = rec_sizes[bi, new_paths[bi] * 3 + 2]

            for u in range(v-2, -1, -1):
                ei_prob = probs_edge[:, u]
                new_edges = ei_prob > 0.5 # broadcast

                for bi, g in enumerate(rec_graphs):
                    if finished[bi]:
                        continue
                    else:
                        # recon edges
                        if new_types[bi] == args['END_TYPE']:
                            ####--- connect it to all loose-end vertices (out_degree==0) ---####
                            no_out_vers = set([u_no_out.index
                                               for u_no_out in g.vs.select(_outdegree_eq=0)
                                               if u_no_out.index != g.vcount() - 1])
                            for u_no_out in no_out_vers:
                                g.add_edge(u_no_out, g.vcount() - 1)

                            finished[bi] = True
                            continue
                        if new_edges[bi, 0]:
                            g.add_edge(u, g.vcount() - 1)

            for i, g in enumerate(rec_graphs):
                utils_data.add_topology_position(g) # add vertice topology order according to the edge
        
        return rec_graphs


    def compute_loss(self, batch):
        """Computes reconstruction loss using teacher forcing.
        
        Computes cross-entropy loss for component types and paths, binary cross-entropy
        for edges, and MSE loss for component sizes. Also tracks accuracy metrics.
        
        Args:
            batch: Dictionary containing:
                - v_types: Ground truth component types.
                - num_nodes: Number of nodes in each graph.
                - gnd_types: Ground truth types for teacher forcing.
                - gnd_paths: Ground truth paths for teacher forcing.
                - gnd_edges: Ground truth edge connectivity.
                - v_sizes: Ground truth component sizes.
                - ckt_latents: Latent circuit representations.
                
        Returns:
            Dictionary with loss values and accuracy metrics:
                - recon: Total reconstruction loss.
                - types: Component type prediction loss.
                - paths: Path prediction loss.
                - edges: Edge prediction loss.
                - sizes: Component size prediction loss.
                - acc_type: Type prediction accuracy.
                - acc_path: Path prediction accuracy.
                - acc_edge: Edge prediction accuracy.
        """
        bsz, n = batch['v_types'].size()

        num_nodes   = batch['num_nodes']
        gnd_types   = batch['gnd_types']
        gnd_paths   = batch['gnd_paths']
        gnd_edges   = batch['gnd_edges']
        gnd_sizes   = batch['v_sizes']

        outputs = self.trans_cross(batch)
        
        rec_sizes = self.size_decoder(batch['ckt_latents']).squeeze(0)

        loss_type, loss_path, loss_edge, loss_size = 0.0, 0.0, 0.0, 0.0
        logits_type, logits_path, logits_topo = (outputs['logits_type'][:, :n, :], 
                                                 outputs['logits_path'][:, :n, :],
                                                 outputs['logits_topo'][:, :n, :])

        loss_ce = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)

        right_types, right_paths, right_edges, tot_ver, tot_edge = 0, 0, 0, 0, 0
        for i in range(bsz):
            # compute acc
            tot_ver += num_nodes[i] - 1 # no start
            probs_type_i = torch.softmax(logits_type[i], dim=-1)
            probs_path_i = torch.softmax(logits_path[i], dim=-1)
            _, new_types = torch.max(probs_type_i, dim=-1) # (bsz, 1)
            _, new_paths = torch.max(probs_path_i, dim=-1) # (bsz, 1)
            
            for v in range(num_nodes[i]-1):
                if new_types[v] == gnd_types[i][v]:
                    right_types += 1
                if new_paths[v] == gnd_paths[i][v]:
                    right_paths += 1

            loss_type += loss_ce(logits_type[i].squeeze(), gnd_types[i].squeeze())
            loss_path += loss_ce(logits_path[i].squeeze(), gnd_paths[i].squeeze())
            
            # edges log likelihood
            num_nodes_i = num_nodes[i]  # remove start node
            num_pot_edges = int(num_nodes_i * (num_nodes_i-1) / 2.0) # max edges for a dag
            logits_edge = torch.zeros(num_pot_edges, 2*self.emb_dim).to(self.device)
            
            cnt = 0
            for idx in range(num_nodes_i-1, 0, -1):
                logits_edge[cnt:cnt+idx, :] = torch.cat([torch.stack([logits_topo[i, idx, :]]*idx, 0), 
                                                         logits_topo[i, :idx, :]], -1)
                cnt += idx

            probs_edge = torch.sigmoid(self.edge_fc(logits_edge))
            loss_edge += F.binary_cross_entropy(probs_edge, gnd_edges[i].to(self.device), reduction='sum')
            
            # compute edge acc
            tot_edge += num_pot_edges
            
            for ei, prob_ei in enumerate(probs_edge):
                if prob_ei > 0.5 and gnd_edges[i][ei]==1:
                    right_edges += 1
                elif prob_ei <= 0.5 and gnd_edges[i][ei]==0:
                    right_edges += 1

        loss_size = self.size_rate*F.mse_loss(rec_sizes, gnd_sizes, reduction='sum') / 300
        rec_loss = self.type_rate*loss_type + self.path_rate*loss_path + loss_edge + loss_size
        
        return {'recon': rec_loss, 'types': loss_type, 'paths': loss_path, "edges": loss_edge, 'sizes': loss_size,
                'acc_type': right_types/tot_ver, 'acc_path': right_paths / tot_ver, 'acc_edge': right_edges / tot_edge}


    # def decode(self, args, batch):


class CausalSelfAttention(nn.Module):
    """Causal self-attention mechanism for autoregressive generation.
    
    Implements multi-head self-attention with causal masking to ensure that
    predictions for position i can only attend to positions <= i.
    
    Args:
        emb_dim: Embedding dimension (must be divisible by 8).
        block_size: Maximum sequence length.
        num_heads: Number of attention heads.
        drop_out_rate: Dropout probability.
    """
    def __init__(self, emb_dim=512, block_size=52, num_heads=8, drop_out_rate=0.1):
        super().__init__()
        assert emb_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(emb_dim, emb_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.num_heads = num_heads

    def forward(self, x):
        """Applies causal self-attention to input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, emb_dim).
        """
        B, T, C = x.size() 
        t = T // 3

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        attn = attn.masked_fill(self.mask[:, :, :t,:t].repeat(1, 1, 3, 3) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """Transformer block with self-attention and feedforward layers.
    
    A single transformer block consisting of causal self-attention followed by
    a feedforward network, with layer normalization and residual connections.
    
    Args:
        emb_dim: Embedding dimension.
        block_size: Maximum sequence length.
        num_heads: Number of attention heads.
        drop_out_rate: Dropout probability.
        fc_rate: Expansion factor for feedforward hidden layer.
    """
    def __init__(self, emb_dim=512, block_size=16, num_heads=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln_self = nn.LayerNorm(emb_dim)
        self.ln_ff = nn.LayerNorm(emb_dim)

        self.self_attn = CausalSelfAttention(emb_dim, block_size, num_heads, drop_out_rate)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, fc_rate * emb_dim),
                                 nn.GELU(),
                                 nn.Linear(fc_rate * emb_dim, emb_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        """Applies transformer block to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, emb_dim).
        """
        x = x + self.self_attn(self.ln_self(x))
        x = x + self.mlp(self.ln_ff(x))
        return x


class TransGPT_Cross(nn.Module):
    """Transformer with cross-attention for circuit generation.
    
    Combines graph embeddings (type, path, topology) with latent circuit representations
    using cross-attention. Uses position embeddings and multiple transformer blocks to
    generate predictions for node types, paths, and topology.
    
    Args:
        max_n: Maximum number of vertices.
        num_types: Number of component types.
        num_paths: Number of signal paths.
        hidden_dim: Hidden state dimension.
        latent_dim: Latent representation dimension.
        emb_dim: Embedding dimension.
        block_size: Maximum sequence length.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        fc_rate: Feedforward expansion factor.
    """
    def __init__(self, max_n=8, num_types=4, num_paths=2, hidden_dim=512, latent_dim=512, emb_dim=512, 
                 block_size=7, num_layers=2, num_heads=8, dropout_rate=0.1, fc_rate=4):

        super().__init__()

        self.max_n = max_n
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.block_size = block_size

        self.type_embed = nn.Embedding(num_types, hidden_dim)
        self.path_embed = nn.Embedding(num_paths, hidden_dim)
        self.topo_embed = GIN(hidden_dim, dropout_rate, max_n) # TODO: drop out rate
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        self.drop = nn.Dropout(dropout_rate)        
        self.type_head = nn.Sequential(nn.Linear(hidden_dim, emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, num_types))

        self.path_head = nn.Sequential(nn.Linear(hidden_dim, emb_dim),
                                      nn.ReLU(),
                                      nn.Linear(emb_dim, num_paths))
        
        self.topo_head = nn.Sequential(nn.Linear(hidden_dim, emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, emb_dim))

        # transformer block
        self.blocks = nn.Sequential(*[Block(hidden_dim, block_size * 3, num_heads, dropout_rate, fc_rate) # block size * 2, include self and cross
                                      for _ in range(num_layers)])
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size * 3, hidden_dim))
        self.apply(self._init_weights)

    def get_block_size(self):
        """Returns the maximum sequence length."""
        return self.block_size

    def _init_weights(self, module):
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, batch):
        """Forward pass to generate logits for types, paths, and topology.
        
        Args:
            batch: Dictionary containing:
                - num_nodes: Number of nodes per graph.
                - v_types: Vertex types.
                - v_paths: Vertex paths.
                - v_topos_1hot: One-hot encoded topology.
                - adj: Adjacency matrices.
                - ckt_latents: Latent circuit representations.
                
        Returns:
            Dictionary with logits for:
                - logits_type: Component type predictions.
                - logits_path: Path predictions.
                - logits_topo: Topology predictions.
        """
        num_nodes, v_types, v_paths, v_topos_1hot, adj, ckt_latents = (batch['num_nodes'],
                                                                       batch['v_types'], 
                                                                       batch['v_paths'], 
                                                                       batch['v_topos_1hot'], 
                                                                       batch['adj'], 
                                                                       batch['ckt_latents'])
        if len(ckt_latents.shape) == 2:
            ckt_latents = ckt_latents.unsqueeze(0)

        b, t = v_paths.shape
        t = t + 1
        assert t <= self.block_size, 'Cannot forward, model block size is exhausted.'
        
        # forward the Trans model
        v_type_embs = self.type_embed(v_types)              # (bs, max_n, emb_dim)
        v_path_embs = self.path_embed(v_paths)              # (bs, max_n, emb_dim)
        v_topo_embs = self.topo_embed(v_topos_1hot, adj)    # (bsz, max_n, max_n) -> (bsz, max_n, node_emb_dim)
        
        ckt_latents = ckt_latents.permute(1, 0, 2) 
        ckt_latents = self.latent_proj(ckt_latents)
        
        g_cat = torch.cat((torch.cat((ckt_latents, v_type_embs), dim=1),
                           torch.cat((ckt_latents, v_path_embs), dim=1),
                           torch.cat((ckt_latents, v_topo_embs), dim=1),), dim=1)

        posi_embs = torch.cat([self.pos_emb[:, :t, :], # cross attn
                               self.pos_emb[:, self.block_size: self.block_size + t, :],
                               self.pos_emb[:, self.block_size + t: self.block_size + 2 * t, :]], dim=1) # self attn
        
        x = self.drop(g_cat + posi_embs)    # (bs, max_n * 2, input_dim)
        x = self.blocks(x)
        x_cat = torch.cat((x[:, :t, :],
                           x[:, t:t*2, :],
                           x[:, t*2:t*3, :],), dim=-1)
        
        logits_type = self.type_head(x[:, :t, :])
        logits_path = self.path_head(x[:, t:t*2, :])
        logits_topo = self.topo_head(x[:, t*2:t*3, :])

        return {'logits_type': logits_type, 'logits_path': logits_path, 'logits_topo': logits_topo}
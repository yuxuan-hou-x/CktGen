"""DIGIN: Lightweight GNN architecture for fast circuit processing.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
import torch.nn as nn


class GINLayer(nn.Module):
    """Graph Isomorphism Network (GIN) layer.
    
    Implements a GIN layer that aggregates neighbor features using a learnable
    epsilon parameter and an MLP for feature transformation.
    
    Args:
        hidden_dim: Hidden dimension for node features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_v, neighbor_sum):
        """Applies GIN layer to node features.
        
        Args:
            h_v: Current node features (batch_size, hidden_dim).
            neighbor_sum: Sum of neighbor features (batch_size, hidden_dim).
            
        Returns:
            Updated node features (batch_size, hidden_dim).
        """
        x = (1 + self.eps) * h_v + neighbor_sum
        return self.mlp(x)

class DIGIN(nn.Module):
    """Lightweight GNN encoder for fast circuit processing.
    
    DIGIN (Direct GIN) is a simplified GNN architecture that processes circuit
    graphs efficiently using GIN layers. It's designed for speed while maintaining
    reasonable accuracy.
    
    Args:
        device: Device to run the model on (CPU/GPU).
        max_n: Maximum number of vertices in a circuit.
        num_types: Number of component types.
        num_paths: Number of signal paths.
        emb_dim: Embedding dimension for features.
        size_emb_dim: Embedding dimension for component sizes.
        hidden_dim: Hidden dimension for GIN layers.
        latent_dim: Dimension of output latent representation.
        **kargs: Additional arguments.
    """
    def __init__(self, device, max_n, num_types, num_paths=8, emb_dim=16, size_emb_dim=8, hidden_dim=64, latent_dim=64, **kargs):

        super().__init__()

        self.max_n = max_n
        self.device = device
        self.hidden_dim = hidden_dim

        self.type_embed = nn.Embedding(num_types, emb_dim)
        self.path_embed = nn.Embedding(num_paths, emb_dim)
        self.size_embed = nn.Sequential(nn.Linear(max_n * 3, emb_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim, size_emb_dim))
        
        self.gin_layer = GINLayer(hidden_dim)
        self.hidden_linear = nn.Linear(emb_dim*2, hidden_dim)
        self.pool_project = nn.Sequential(
            nn.Linear(hidden_dim*self.max_n, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim),
        )
        self.g_project = nn.Linear(hidden_dim+size_emb_dim, latent_dim)

    def propagate_to(self, G, v):
        """Propagates information to vertex v in all graphs.
        
        Args:
            G: List of igraph circuit graphs.
            v: Vertex index to propagate to.
        """
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0: return
        
        # make one-hot for v type and position 
        v_types = torch.tensor([g.vs[v]['type'] for g in G]).to(self.device)                     # vertice v type list
        v_paths = torch.tensor([g.vs[v]['path'] for g in G]).to(self.device)                     # vertice v position list 

        type_embs = self.type_embed(v_types)
        path_embs = self.path_embed(v_paths)

        h_v = torch.cat([type_embs, path_embs], dim=-1)             # concate v type and position, [subg_type_onehot, position_onehot]
        h_v = self.hidden_linear(h_v)
        # h_v = torch.cat([g.vs[v].get('H', self._get_zero_hidden(1)) for g in G], 0)

        # 求和所有前驱节点 hidden
        neighbor_sum = []
        for g in G:
            preds = g.predecessors(v)
            if preds:
                sum_h = torch.sum(
                torch.cat(
                    [g.vs[u]['h']
                    for u in preds], 
                    0), 
                dim=0, 
                keepdim=True
            )
            else:
                sum_h = torch.zeros(1, self.hidden_dim).to(self.device) #self._get_zero_hidden(1)

            neighbor_sum.append(sum_h)


        neighbor_sum = torch.cat(neighbor_sum, 0)
        # print('h_v size: ', h_v.size())

        # GIN 层
        h_new = self.gin_layer(h_v, neighbor_sum)
        # print('h_new size: ', h_new.size())
        for i, g in enumerate(G):
            g.vs[v]['h'] = h_new[i:i+1]
        return h_new

    def get_pooling(self, G):
        """Pools node features into a graph-level representation.
        
        Collects node features from all vertices in each graph, pads to max_n vertices,
        flattens the features, and projects them to a fixed-dimensional graph embedding.
        
        Args:
            G: List of igraph circuit graphs with node features stored in 'h' attribute.
            
        Returns:
            Graph embeddings of shape (batch_size, hidden_dim).
        """
        H_batch = []
        for g in G:
            H = [torch.as_tensor(g.vs[v]['h'], device=self.device) for v in range(g.vcount())]
            if g.vcount() < self.max_n:
                H += [torch.zeros(1, self.hidden_dim, device=self.device)] * (self.max_n - g.vcount())

            H = torch.stack(H, dim=0)              # [max_len, h_dim]
            H_flatten = H.reshape(1, -1)           # [1, max_len * h_dim]
            H_batch.append(H_flatten)

        H_batch = torch.cat(H_batch, dim=0)        # [batch_size, max_len * h_dim]

        g_embs = self.pool_project(H_batch)        # [batch_size, out_dim]
        return g_embs


    def forward(self, batch):
        """Encodes circuit graphs into latent representations.
        
        Processes circuit graphs through GIN layers in topological order, pools the
        node features, concatenates with size embeddings, and projects to latent space.
        
        Args:
            batch: Dictionary containing:
                - 'G': Circuit graph(s) (igraph object or list of igraph objects).
                - 'v_sizes': Component size information (batch_size, max_n * 3).
                
        Returns:
            Circuit latent embeddings of shape (batch_size, latent_dim).
        """
        G = batch['G']

        if type(G) != list:
            G = [G]

        prop_order = range(0, self.max_n)                               # make a vector, [], v: start, self.max_n: stop, 1: step

        for u in prop_order:
            self.propagate_to(G=G, v=u)

        g_embs = self.get_pooling(G)


        v_sizes = batch['v_sizes']
        size_embs = self.size_embed(v_sizes)

        g_size_embs = torch.cat([g_embs, size_embs], dim=-1)
        g_embs = self.g_project(g_size_embs)

        return g_embs
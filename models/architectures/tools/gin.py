"""Graph Isomorphism Network (GIN) layers for circuit graphs.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
from torch import nn


class GIN(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_n=20, device=None):
        super(GIN, self).__init__()

        self.emb_dim = emb_dim  # size of the position embedding
        self.max_n = max_n  # maximum position
        self.dropout = dropout

        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)

        self.W1 = nn.Parameter(torch.zeros(2 * max_n, 2 * emb_dim))
        self.W2 = nn.Parameter(torch.zeros(2 * emb_dim, emb_dim))

        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.relu = nn.ReLU()
        self.max_n = max_n
        self.device = device
 
    def forward(self, x, adj):
        """
            x is the postiion list, size = (batch, max_n, max_n):
            one-hot of position, and nodes after the end type are all zeros embedding
            adj is the adjacency matrix (not the sparse matrix)

            # bsize = len(x)
            pos_one_hot = torch.zeros(bsize, self.max_n, self.max_n).to(self._get_device())
            for i in range(bsize):
            pos_one_hot[i, :len(x[i]), :] = self._one_hot(x[i], self.max_n)
        """

        # transpose adj, a row of adj is the out edge, a column of adj is the in edge
        pos_embed = torch.cat((x, torch.matmul(adj.transpose(1, 2), x)), 2)  # popagate the position information
        pos_embed = self.relu(torch.matmul(pos_embed, self.W1.to(self._get_device())))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        pos_embed = torch.matmul(pos_embed, self.W2.to(self._get_device()))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        return pos_embed

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
"""CktGNN: Graph Neural Network architecture for circuit encoding.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import numpy as np
import igraph
import torch
import torch.nn as nn
import torch.nn.functional as F


# Some utility functions

# subgraph basis type, that is node typwe
SUBG_NODE = {0: ['In'],
            1: ['Out'],
            2: ['R'],
            3: ['C'],
            4: ['R', 'C'],
            5: ['R', 'C'],
            6: ['+gm+'],
            7: ['-gm+'],
            8: ['+gm-'],
            9: ['-gm-'],
            10: ['C', '+gm+'],
            11: ['C', '-gm+'],
            12: ['C', '+gm-'],
            13: ['C', '-gm-'],
            14: ['R', '+gm+'],
            15: ['R', '-gm+'],
            16: ['R', '+gm-'],
            17: ['R', '-gm-'],
            18: ['C', 'R', '+gm+'],
            19: ['C', 'R', '-gm+'],
            20: ['C', 'R', '+gm-'],
            21: ['C', 'R', '-gm-'],
            22: ['C', 'R', '+gm+'],
            23: ['C', 'R', '-gm+'],
            24: ['C', 'R', '+gm-'],
            25: ['C', 'R', '-gm-']}

# subgraph connection way for each node
SUBG_CON = {0: None,
            1: None,
            2: None,
            3: None,
            4: 'series',
            5: 'parral',
            6: None,
            7: None,
            8: None,
            9: None,
            10: 'parral',
            11: 'parral',
            12: 'parral',
            13: 'parral',
            14: 'parral',
            15: 'parral',
            16: 'parral',
            17: 'parral',
            18: 'parral',
            19: 'parral',
            20: 'parral',
            21: 'parral',
            22: 'series',
            23: 'series',
            24: 'series',
            25: 'series'}

SUBG_INDI = {0: [],  
             1: [],      
             2: [0],     
             3: [1],     
             4: [0, 1],  
             5: [0, 1],  
             6: [2],     
             7: [2],     
             8: [2], 
             9: [2], 
             10: [1, 2], 
             11: [1, 2], 
             12: [1, 2], 
             13: [1, 2], 
             14: [0, 2], 
             15: [0, 2], 
             16: [0, 2], 
             17: [0, 2],
             18: [1, 0, 2],  
             19: [1, 0, 2],  
             20: [1, 0, 2],  
             21: [1, 0, 2],  
             22: [1, 0, 2],  
             23: [1, 0, 2],
             24: [1, 0, 2],
             25: [1, 0, 2]}


def one_hot(idx, length):
    """make a one-hot matrix of length and set idx-th element to be 1"""
    if type(idx) in [list, range]:
        if idx == []:
            return None
        idx = torch.LongTensor(idx).unsqueeze(0).t()
        x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    else:
        idx = torch.LongTensor([idx]).unsqueeze(0)
        x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


class CKTGNN(nn.Module):
    # topology and node feature together.
    def __init__(self, max_n, num_types, START_TYPE, END_TYPE, max_pos=8, emb_dim=16, feat_emb_dim=8, hid_dim=301, latent_dim=56,
                 bidirectional=False, sized=True, topo_feat_scale=0.01, **kargs):

        super(CKTGNN, self).__init__()

        self.max_n = max_n  # maximum number of vertices
        self.max_pos = max_pos + 1  # number of positions in amp: 1 sudo + 7 positions
        self.num_types = num_types  # number of device types
        self.START_TYPE = START_TYPE  # 0 for start
        self.END_TYPE = END_TYPE  # 1 for end
        self.emb_dim = emb_dim  # embedding dimension
        self.feat_emb_dim = feat_emb_dim  # continuous feature embedding dimension
        self.hid_dim = hid_dim  # hidden state size of each vertex
        self.latent_dim = latent_dim  # size of latent representation z
        self.g_emb_sz = hid_dim + feat_emb_dim  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.sized = sized  # whether to use the position information
        self.topo_feat_scale = topo_feat_scale # balance the attntion to topology loss
        self.device = None

        if self.sized:
            self.v_stat_sz = hid_dim + self.max_pos
        else:
            self.v_stat_sz = hid_dim

        # 0. encoding-related
        #   device parameter feature mlp
        self.df_enc = nn.Sequential(nn.Linear(self.max_pos * 3, emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(emb_dim, feat_emb_dim))  # subg features can be canonized according to the position of subg TODO: what does it mean

        self.grue_forward = nn.GRUCell(num_types + self.max_pos, hid_dim)
        self.grue_backward = nn.GRUCell(num_types + self.max_pos, hid_dim)  # backward encoder
        self.fc1 = nn.Linear(self.g_emb_sz, latent_dim)  # predict latent mean
        self.fc2 = nn.Linear(self.g_emb_sz, latent_dim)  # predict latent logvar

        # 1. decoding-related
        self.grud = nn.GRUCell(num_types + self.max_pos, hid_dim)  # decoder GRU
        self.fc3 = nn.Linear(latent_dim, hid_dim)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), 
                                        nn.ReLU(), 
                                        nn.Linear(hid_dim * 2, num_types))  # predict which type of new subg to add
        self.add_edge = nn.Sequential(nn.Linear(hid_dim * 2 + self.max_pos * 2, hid_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(hid_dim * 4, 1))  # predict whether to add edge between v_i and v_new

        self.add_pos = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(hid_dim * 2, self.max_pos))  # which position of new subg to add, predict add node position

        self.df_fc = nn.Sequential(nn.Linear(hid_dim, 64), 
                                   nn.ReLU(), 
                                   nn.Linear(64, self.max_pos * 3))  # decode subg features

        # 2. gate-related
        self.gate_forward = nn.Sequential(nn.Linear(self.v_stat_sz, hid_dim), 
                                          nn.Sigmoid())
        self.gate_backward = nn.Sequential(nn.Linear(self.v_stat_sz, hid_dim), 
                                           nn.Sigmoid())
        self.mapper_forward = nn.Linear(self.v_stat_sz, hid_dim, bias=False) # map v state to hidden state
        # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Linear(self.v_stat_sz, hid_dim, bias=False)

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Linear(hid_dim * 2, hid_dim)

            """ make two concate hidden state size to one size use MLP """
            self.hg_unify = nn.Linear(self.g_emb_sz * 2, self.g_emb_sz)

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)


    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device


    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device())  # get a zero hidden state


    def _get_zero_hidden(self, n=1, prior_edge=False):
        if prior_edge:
            return self._get_zeros(n, self.hid_dim + self.max_pos)
        else:
            return self._get_zeros(n, self.hid_dim)


    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()  # make it a column tensor
            x = (
                torch.zeros((len(idx), length))
                .scatter_(1, idx, 1)  # set column in idx to be 1, rest be 0
                .to(self.get_device())
            )
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x


    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)


    def _collate_fn(self, G):
        return [g.copy() for g in G]


    def _propagate_to(self, G, v, propagator, H=None, reverse=False, decode=False):
        # G is a batch of graph
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]

        if len(G) == 0:
            return

        if H is not None:  # H: previous hidden state
            idx = [i for i, g in enumerate(G) if g.vcount() > v]    # graph index of a batch which vcount > v
            H = H[idx]                                              # hidden state sets for each graph

        # make one-hot for v type and position
        v_types = [g.vs[v]['type'] for g in G]                      # vertice v type list
        pos_feats = [g.vs[v]['path'] for g in G]                     # vertice v position list 
        X_v_ = self._one_hot(v_types, self.num_types)                     # encode v type as one-hot (G.size, num_types)
        X_pos_ = self._one_hot(pos_feats, self.max_pos)             # encode v position{node index} as one-hot 
        X = torch.cat([X_v_, X_pos_], dim=1)                        # concate v type and position, [subg_type_onehot, position_onehot]
        
        # print(X)
        
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] 
                       for g in G]  # hidden state of 'predecessors', init v's successors hidden state
            
            if self.sized:
                pos_ = [self._one_hot([g.vs[v_]['path'] for v_ in g.successors(v)], self.max_pos) for g in G]  # one hot of vertex index of 'predecessors', in reverse situation, successors is the pre node, pos_ = vids
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]          # Hidden state for pre nodes
            if self.sized:
                pos_ = [self._one_hot([g.vs[x]['path'] for x in g.predecessors(v)], self.max_pos) for g in G]  # one hot of vertex index of 'predecessors', pos_=vids
            gate, mapper = self.gate_forward, self.mapper_forward
        
        if self.sized:                                                # 
            H_pred = [[torch.cat([x[i], y[i : i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, pos_)]
        # print(H_pred)
        
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])              # maximum number of predecessors
            if max_n_pred == 0:                                     ### start point
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + [self._get_zeros(max_n_pred - len(h_pred), self.v_stat_sz)], 0).unsqueeze(0)
                          for h_pred in H_pred]                     # pad all to same length
                H_pred = torch.cat(H_pred, 0)                       # batch * max_n_pred * v_stat_sz
                H = self._gated(H_pred, gate, mapper).sum(1)        # batch * hid_dim
                
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i : i + 1]
        return Hv


    def _propagate_from(self, G, v, propagator, H0=None, reverse=False, decode=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)                                   # make a vector, [v, v-1, ..., 0], v: start, -1: stop, -1: step
        else:
            prop_order = range(v, self.max_n)                               # make a vector, [], v: start, self.max_n: stop, 1: step

        Hv = self._propagate_to(
            G, v, propagator, H0, reverse=reverse, decode=decode
        )  # the initial vertex, initial the vertex state
        
        for v_ in prop_order[1:]:
            """prop according to the top order"""
            self._propagate_to(G, v_, propagator, reverse=reverse, decode=decode)
            # Hv = self._propagate_to(G, v_, propagator, Hv, reverse=reverse, decode=decode)
            # Hv = self._propagate_to(G, v_, propagator, Hv, reverse=reverse) no need
        return Hv


    def _update_v(self, G, v, H0=None, decode=False):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False, decode=decode)
        return


    def _get_vertex_state(self, G, v, prior_edge=False):
        # get the vertex states at v
        # prior_edge indicates whether to include the edge information from v's predecessors
        Hv = []
        for g in G:
            if v >= g.vcount():  # TODO: prior_edge means Sudo node?
                hv = self._get_zero_hidden(prior_edge=prior_edge)
            else:
                hv = g.vs[v]['H_forward']
                if prior_edge:
                    pos_ = self._one_hot([g.vs[v]['path']], self.max_pos)
                    hv = torch.cat([hv, pos_], 1)
            Hv.append(hv)
        Hv = torch.cat(Hv, 0) # make tensor in a matrix
        return Hv


    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount() - 1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward encode
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)  # a linear model to make bidirection propagate to same size like single directions

        return Hg


    def encode(self, batch):
        G = batch['G']
        # encode graphs G into latent vectors
        # G is a batch set of graphs
        if type(G) != list:
            G = [G]

        """ Initialize the hidden state with empty list each graph in a batch """
        self._propagate_from(G, 0, self.grue_forward,
                             H0=self._get_zero_hidden(len(G)),
                             reverse=False, decode=False)

        if self.bidir:
            self._propagate_from(G, 
                                 self.max_n - 1,
                                 self.grue_backward,
                                 H0=self._get_zero_hidden(len(G)),
                                 reverse=True, decode=False)

        Hg = self._get_graph_state(G) # make reverse and forword into hid_dim
        # print(Hg.shape)

        dfs_ = []
        for g in G:
            df_ = [0] * (3 * self.max_pos)

            for v_ in range(len(g.vs)):
                pos_ = g.vs[v_]['path']
                df_[pos_ * 3 + 0] = g.vs[v_]['r']  # r value in subg
                df_[pos_ * 3 + 1] = g.vs[v_]['c']  # c value in subg
                df_[pos_ * 3 + 2] = g.vs[v_]['gm']  # gm value in subg
            dfs_.append(df_)

        Hdf = torch.FloatTensor(dfs_).to(self.get_device())  # converts to a tensor
        Hd = self.df_enc(Hdf)  # encode to a low dim embedding
        # print('before concate : {}'.format(Hd.shape))
        Hg = torch.cat([Hg, Hd], dim=1)  #  concatenate the topology embedding and subg feature embedding
        # print('after concate : {}'.format(Hg.shape))
        mu, logvar = self.fc1(Hg), self.fc2(Hg)

        return {'ckt_dists': (mu, logvar)}


    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))


    def decode(self, args, batch, stochastic=True):
        ckt_latents = batch['ckt_latents']
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(ckt_latents))  # or relu activation, similar performance
        pred_dfs = self.df_fc(H0)  # hidden to features


        G = [igraph.Graph(directed=True) for _ in range(len(ckt_latents))]  # make batch size empty igraph data
        for g in G:
            """add In node in each graph"""
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['r'] = 0.0
            g.vs[0]['c'] = 0.0
            g.vs[0]['gm'] = 0.0
            g.vs[0]['path'] = 0

        self._update_v(G, 0, H0, decode=True)  
        finished = [False] * len(G)  

        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True) # the last hidden state which contain previous info  

                type_scores = self.add_vertex(Hg)  # predict the type of the next added vertex

                pos_scores = self.add_pos(Hg)  # predict the position of the next added vertex
                
                # pred_dfs = self.df_fc(Hg)

                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    pos_probs = F.softmax(pos_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.num_types), p=type_probs[i])
                                 for i in range(len(G))]
                    new_pos = [np.random.choice(range(self.max_pos), p=pos_probs[i])
                               for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
                    new_pos = torch.argmax(pos_scores, 1)
                    new_pos = new_pos.flatten().tolist()

            for j, g in enumerate(G):
                if not finished[j]:
                    g.add_vertex(type=new_types[j])
                    g.vs[idx]['path'] = new_pos[j]
                    g.vs[idx]['r'] = pred_dfs[j, new_pos[j] * 3 + 0]
                    g.vs[idx]['c'] = pred_dfs[j, new_pos[j] * 3 + 1]
                    g.vs[idx]['gm'] = pred_dfs[j, new_pos[j] * 3 + 2]

            # TODO: how to deal with multi previous nodes
            self._update_v(G, idx, decode=True)

            # decide connections
            edge_scores = []
            for vi in range(idx - 1, -1, -1):  # vi in [idx - 1, 0]
                Hvi = self._get_vertex_state(G, vi, prior_edge=True)
                H = self._get_vertex_state(G, idx, prior_edge=True)
                ei_score = self._get_edge_score(Hvi, H, H0)  # H0: current graph state

                # wheather the threshold is random
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5

                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE:
                        # if new node is end_type, connect it to all loose-end vertices (out_degree == 0)
                        end_vertices = set([v.index
                                            for v in g.vs.select(_outdegree_eq=0)
                                            if v.index != g.vcount() - 1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount() - 1)
                            
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount() - 1)
                self._update_v(G, idx, decode=True)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory

        return G


    def compute_loss(self, batch):
        """
            compute the loss of decoding mu and logvar to true graphs using teacher forcing
            ensure when computing the loss of step i, steps 0 to i - 1 are correct
        """
        ckt_latents, G_true = batch['ckt_latents'], batch['G']


        dfs_ = []
        for g in G_true:
            df_ = [0] * (3 * self.max_pos)
            for v_ in range(len(g.vs)):
                pos_ = g.vs[v_]['path']
                df_[pos_ * 3 + 0] = g.vs[v_]['r']
                df_[pos_ * 3 + 1] = g.vs[v_]['c']
                df_[pos_ * 3 + 2] = g.vs[v_]['gm']
            dfs_.append(df_)
        true_dfs = torch.FloatTensor(dfs_).to(self.get_device())
        
        H0 = self.tanh(self.fc3(ckt_latents))  # or relu activation, similar performance
        pred_dfs = self.df_fc(H0)

        G = [igraph.Graph(directed=True) for _ in range(len(ckt_latents))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['r'] = 0.0
            g.vs[0]['c'] = 0.0
            g.vs[0]['gm'] = 0.0
            g.vs[0]['path'] = 0

        self._update_v(G, 0, H0)

        res = 0  # log likelihood
        total_type_loss, total_pos_loss, total_edge_loss = 0.0, 0.0, 0.0
        
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type']
                          if v_true < g_true.vcount()  # (bsize, 1)
                          else self.START_TYPE # use start type to denote padding vertices since start type only appears for vertex 0
                          for g_true in G_true]
            
            true_pos = [g_true.vs[v_true]['path']
                        if v_true < g_true.vcount()  # (bsize, 1)
                        else self.max_pos - 1 # 
                        for g_true in G_true]

            Hg = self._get_graph_state(G, decode=True) # get every node hidden state and stored in memory

            type_scores = self.add_vertex(Hg)  # (bsize, self.vrt)
            pos_scores = self.add_pos(Hg)

            # vertex log likelihood
            v_type_logsoftmax = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum() # compute the true type score
            v_pos_logsoftmax = self.logsoftmax1(pos_scores)[np.arange(len(G)), true_pos].sum()

            res = res + v_type_logsoftmax + v_pos_logsoftmax

            total_type_loss += v_type_logsoftmax
            total_pos_loss += v_pos_logsoftmax

            # add true node not the predict node
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
                    g.vs[v_true]['r'] = G_true[i].vs[v_true]['r']
                    g.vs[v_true]['c'] = G_true[i].vs[v_true]['c']
                    g.vs[v_true]['gm'] = G_true[i].vs[v_true]['gm']
                    g.vs[v_true]['path'] = G_true[i].vs[v_true]['path']
            
            self._update_v(G, v_true) # to init the unexisted hidden state of v_true

            H = self._get_vertex_state(G, v_true)
            true_edges = []
            
            # get_idjlist: 
            #   return a list of node index to show these directed edges. true_edges[i] = in ith graph, v_true's predecessors

            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true]
                                  if v_true < g_true.vcount() # 
                                  else []) # ignore the padding nodes
                    
            edge_scores = []

            # use true node as present node
            for vi in range(v_true - 1, -1, -1):
                Hvi = self._get_vertex_state(G, vi, prior_edge=True)
                H = self._get_vertex_state(G, v_true, prior_edge=True)
                ei_score = self._get_edge_score(Hvi, H, H0)  # size: batch size, 1
                edge_scores.append(ei_score)
                for i, g in enumerate(G):  # only add true edge
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true)

            # (batch size, v_true): columns: v_true-1, ..., 0
            edge_scores = torch.cat(edge_scores[::-1], 1)  

            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0

            # edges log-likelihood
            edge_loss = -F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum')
            total_edge_loss += edge_loss

            res = res + edge_loss

            # TODO: if add node is end, end the loop?

        # feature loss
         # each subg node has 3 subg features, which scan be normalized by divide 100.
        

        res1 = -res  # convert likelihood to loss
        total_type_loss = -total_type_loss
        total_pos_loss = -total_pos_loss

        
        res_v_feat_log = (self.topo_feat_scale * F.mse_loss(pred_dfs, true_dfs, reduction='sum') / 300) 
        res = res1 + res_v_feat_log


        return {'recon': res1, 'types': total_type_loss, 'paths': total_pos_loss,
                'sizes': res_v_feat_log, 'edges': -total_edge_loss, 'acc_type': 0, 'acc_path': 0}

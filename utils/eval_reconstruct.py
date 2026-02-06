"""Reconstruction evaluation utilities.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import igraph
import torch
import numpy as np
from torch.nn import functional as F
import utils.data as utils_data
# import pandas as pd


def init_recon_graph(args, bsz):
    """Initializes a batch of empty graphs for reconstruction.
    
    Creates directed graphs with a single start vertex (type=START_TYPE) initialized
    with default device parameters. Used for autoregressive circuit generation.
    
    Args:
        args: Configuration dictionary with 'START_TYPE' key.
        bsz: Batch size - number of graphs to initialize.
        
    Returns:
        list: List of bsz igraph.Graph objects, each with one initialized vertex.
        
    Notes:
        Each graph's first vertex has:
        - type: args['START_TYPE'] (typically 0 for input node)
        - topo: 0 (topological position)
        - path: 0 (path index)
        - r, c, gm: 0 (device parameters)
    """
    G = [igraph.Graph(directed=True) for _ in range(bsz)]
    for g in G:
        g.add_vertex(type=args['START_TYPE'])
        g.vs[0]['topo'] = 0
        g.vs[0]['path'] = 0
        g.vs[0]['r'] = 0
        g.vs[0]['c'] = 0
        g.vs[0]['gm'] = 0

    return G

def init_recon_graph_with_start(args, bsz):
    """Initializes a batch of graphs with explicit start symbol for PACE architecture.
    
    Creates directed graphs with two vertices: a special START_SYMBOL node (vertex 0)
    and a START_TYPE node (vertex 1), connected by an edge. Used for autoregressive
    generation in PACE which requires explicit start symbols.
    
    Args:
        args: Configuration dictionary with 'START_SYMBOL' and 'START_TYPE' keys.
        bsz: Batch size - number of graphs to initialize.
        
    Returns:
        list: List of bsz igraph.Graph objects, each with two connected vertices.
        
    Notes:
        Each graph has:
        - Vertex 0 (start symbol): type=START_SYMBOL, path=2, topo=0
        - Vertex 1 (start node): type=START_TYPE, path=0, topo=1
        - Edge 0->1
        - All r, c, gm parameters initialized to 0
    """
    G = [igraph.Graph(directed=True) for _ in range(bsz)]
    for g in G:
        g.add_vertex(type=args['START_SYMBOL'])
        g.vs[0]['topo'] = 0
        g.vs[0]['path'] = 2
        g.vs[0]['r'] = 0
        g.vs[0]['c'] = 0
        g.vs[0]['gm'] = 0

        g.add_vertex(type=args['START_TYPE'])
        g.vs[1]['topo'] = 1
        g.vs[1]['path'] = 0
        g.vs[1]['r'] = 0
        g.vs[1]['c'] = 0
        g.vs[1]['gm'] = 0
        
        g.add_edge(0, 1)

    return G


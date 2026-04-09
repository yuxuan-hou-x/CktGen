"""Reconstruction evaluation utilities.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""

import igraph


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

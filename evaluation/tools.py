"""Utility functions for evaluation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import time
import torch
import numpy as np
import utils.data as utils_data
from tqdm import tqdm


def compute_retrieval_precision(spec_latents, ckt_latents, gnds):
    assert len(ckt_latents) == len(spec_latents)
    batch_size, _ = spec_latents.shape
    # nomalize
    ckt_latents = torch.nn.functional.normalize(ckt_latents, dim=-1).squeeze()
    spec_latents = torch.nn.functional.normalize(spec_latents, dim=-1).squeeze()

    sim_matrix = ckt_latents @ spec_latents.T    # (b, d) x (d, b) -> (b, b)
    sim_scores, indices = torch.sort(sim_matrix, dim=1, descending=True)

    ranks = []
    for i, gnd in enumerate(gnds):
        rank = (indices[i] == gnd).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks)
    
    top1  = torch.mean((ranks == 1).float()).item()
    top2  = torch.mean((ranks <= 2).float()).item()
    top3  = torch.mean((ranks <= 3).float()).item()
    top5  = torch.mean((ranks <= 5).float()).item()
    top10 = torch.mean((ranks <= 10).float()).item()
    avg   = torch.mean(ranks.float()).item()

    return {"Top1": top1, "Top2": top2, "Top3": top3, "Top5": top5, "Top10": top10, 'avg': avg}

def euclidean_distance_matrix(matrix1, matrix2):
    """Computes pairwise Euclidean distances between two sets of vectors.
    
    Efficiently calculates all pairwise L2 distances using vectorized operations
    instead of nested loops. Uses the identity:
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        matrix1: Numpy array of shape (N1, D) - first set of vectors.
        matrix2: Numpy array of shape (N2, D) - second set of vectors.
        
    Returns:
        np.ndarray: Distance matrix of shape (N1, N2) where
                   dists[i, j] = ||matrix1[i] - matrix2[j]||_2
                   
    Raises:
        AssertionError: If feature dimensions don't match.
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def ratio_same_DAG(G0, G1):
    """Calculates the ratio of graphs in G1 that also appear in G0.
    
    Useful for measuring novelty: if comparing generated circuits (G1) against
    training set (G0), a lower ratio indicates more novel/diverse generation.
    
    Args:
        G0: Reference graph list (e.g., training set).
        G1: Query graph list (e.g., generated circuits).
        
    Returns:
        float: Ratio of G1 graphs found in G0, in range [0, 1].
               Lower values indicate more novelty.
               
    Notes:
        - Uses is_same_DAG for exact graph matching
        - O(|G0| * |G1|) complexity - can be slow for large datasets
    """
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1, desc="Comparing graphs"):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
                
    return res / len(G1)


def is_valid_DAG(g, start_symbol=False):
    """Checks if a graph is a valid directed acyclic graph for circuit representation.
    
    A valid DAG must satisfy:
    1. No directed cycles (is_dag property)
    2. Exactly one START node (type 0)
    3. Exactly one END node (type 1)
    4. Optionally one START_SYMBOL node (type 2) if start_symbol=True
    5. No zero-indegree nodes except START (or START_SYMBOL)
    6. No zero-outdegree nodes except END
    
    Args:
        g: igraph.Graph object with 'type' vertex attribute.
        start_symbol: If True, expects a START_SYMBOL node (type 2) in addition
                     to START and END nodes. Default False.
                     
    Returns:
        bool: True if graph is a valid DAG meeting all criteria.
        
    Notes:
        - START_TYPE = 0, END_TYPE = 1, START_SYMBOL = 2
        - Ensures all nodes are properly connected in the computation graph
    """
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node

    START_TYPE = 0
    END_TYPE = 1

    if start_symbol:
        START_SYMBOL = 2

    res = g.is_dag()

    n_start, n_end, n_symbol = 0, 0, 0

    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        elif start_symbol and v['type'] == START_SYMBOL:
            n_symbol += 1

        if start_symbol and v.indegree() == 0 and v['type'] != START_SYMBOL:
            return False
        if not start_symbol and v.indegree() == 0 and v['type'] != START_TYPE:
            return False
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    
    if start_symbol:
        return res and n_start == 1 and n_end == 1 and n_symbol == 1
    else:
        return res and n_start == 1 and n_end == 1


def is_valid_Circuit(g, start_symbol=False):
    """Checks if a graph represents a valid amplifier circuit.
    
    A valid circuit must:
    1. Be a valid DAG (topology check)
    2. Have correct node types in the main signal path (no resistors/capacitors)
    
    Args:
        g: igraph.Graph with 'type' and 'path' vertex attributes.
        start_symbol: If True, uses adjusted path indices [3,4,5] for main path.
                     If False, uses path indices [2,3,4]. Default False.
                     
    Returns:
        bool: True if graph is a valid amplifier circuit.
        
    Notes:
        Main path constraint:
        - If start_symbol=False: path [2,3,4] cannot have types [8,9] (R/C)
        - If start_symbol=True: path [3,4,5] cannot have types [9,10] (R/C)
        This ensures transistors/active elements in signal path.
    """
    # Check if the given igraph g is a amp circuits
    # first checks whether the circuit topology is a DAG
    # second checks the node type in the main path
    cond1 = is_valid_DAG(g, start_symbol)
    cond2 = True

    for v in g.vs:
        path = v['path']

        if start_symbol:
            if path in [3, 4, 5]:  # i.e. in the main path
                if v['type'] in [9, 10]:
                    cond2 = False
        else:
            if path in [2, 3, 4]:  # i.e. in the main path
                if v['type'] in [8, 9]:
                    cond2 = False

    return cond1 and cond2


def is_same_DAG(g0, g1):
    """Checks if two graphs are identical (same structure and node types).
    
    Two graphs are considered the same if:
    1. Same number of vertices
    2. Same node type at each position
    3. Same incoming edges for each node
    
    Args:
        g0: First igraph.Graph with 'type' vertex attribute.
        g1: Second igraph.Graph with 'type' vertex attribute.
        
    Returns:
        bool: True if graphs are structurally identical.
        
    Notes:
        - Does NOT check for graph isomorphism (different node orderings)
        - Assumes node ordering is canonical and consistent
        - Only checks incoming edges (sufficient for DAGs)
    """
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):  # compare the vertice set of in neighbors between generate and gnd-truth
            return False
    return True


@torch.no_grad()
def extract_latents(args, model, data):
    """Extracts mean latent representations from a dataset using a VAE encoder.
    
    Encodes all circuits in the dataset to their latent space representations
    (mean vectors from the VAE's encoder distribution).
    
    Args:
        args: Configuration dictionary for data transformation.
        model: VAE model with archi.encode method.
        data: List of circuit graphs (igraph.Graph objects).
        
    Returns:
        np.ndarray: Concatenated latent vectors of shape (N, latent_dim),
                   where N is the number of circuits.
                   
    Notes:
        - Uses mean of latent distribution (mu), ignores variance
        - Handles both 2D and 3D mu tensors (squeezes if needed)
        - WARNING: Current code has bug - 'latents' undefined, should be 'mu'
    """
    model.eval()

    g_batch = utils_data.collate_fn(data)
    batch = utils_data.transforms(args, g_batch)

    batch.update(model.archi.encode(batch))
    mu, _ = batch['ckt_dists']
    if len(mu.shape) == 3:
        mu = mu.squeeze(0)
    mu = mu.cpu().detach().numpy()

    return mu
"""Data processing utilities for circuit graphs.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import igraph
import torch
import numpy as np
import math
import utils.data as utils_data
from torch.nn import functional as F

# import pandas as pd

#####################################################################################
#                           prepare the data for the model                          #
#####################################################################################
def get_one_hot(ori_lst, num_cls, max_len=None, pad_val=None):
    """Converts integer sequences to one-hot encoded tensors.
    
    Args:
        ori_lst: List of integer sequences or tensor to be one-hot encoded.
        num_cls: Number of classes for one-hot encoding.
        max_len: Maximum sequence length for padding (optional).
        pad_val: Padding value to use if sequences are shorter than max_len (optional).
        
    Returns:
        torch.FloatTensor: One-hot encoded tensor of shape (..., num_cls).
        
    Examples:
        >>> get_one_hot([[0, 1], [2]], num_cls=3, max_len=2, pad_val=0)
        # Returns padded one-hot encoded tensor
    """
    if pad_val is not None:
        pad_lst = [sub_lst + [pad_val] * (max_len - len(sub_lst))
                   for sub_lst in ori_lst]
        indices = torch.tensor(pad_lst)
    else:
        indices = ori_lst
    # 使用 F.one_hot 进行 one-hot 编码
    one_hot_encoded = F.one_hot(indices, num_classes=num_cls).to(dtype=torch.float32)
    return one_hot_encoded


def seq_padding(ori_lst, max_len=None, pad_val=None):
    """Pads a sequence to a specified maximum length.
    
    Args:
        ori_lst: Original list to be padded.
        max_len: Target maximum length.
        pad_val: Value to use for padding.
        
    Returns:
        list: Padded list of length max_len, or original list if already >= max_len.
    """
    if len(ori_lst) < max_len:
        pad_lst = ori_lst + [pad_val] * (max_len - len(ori_lst))
        return pad_lst
    return ori_lst


def collate_fn(G):
    """Collates a batch of graphs by creating copies.
    
    Args:
        G: List of igraph Graph objects.
        
    Returns:
        list: List of deep copies of the input graphs.
    """
    return [g.copy() for g in G]


def add_conditions(datasets, perform_df, start_idx=0):
    """Adds performance conditions (gain, pm, bw, fom) to circuit graphs from dataframe.
    
    Reads performance metrics from a pandas DataFrame and attaches them to circuit graphs.
    If a circuit is marked as invalid, all metrics are set to 0.
    
    Args:
        datasets: List of circuit graph objects to annotate.
        perform_df: Pandas DataFrame with columns ['valid', 'fom', 'gain', 'pm', 'bw'].
        start_idx: Starting index in perform_df for reading data. Default 0.
        
    Notes:
        Performance ranges:
        - gain: max 4.0, min 1.0
        - pm: max 6.0, min -2.0
        - bw: max 32.0, min 0.0
        - fom: varies
        
    Side Effects:
        Modifies graphs in-place by adding 'fom', 'gain', 'pm', 'bw' attributes.
    """
    for i in range(len(datasets)):
        g = datasets[i]
        if perform_df['valid'][start_idx + i] == 1:
            fom = perform_df['fom'][start_idx + i]
            gain = perform_df['gain'][start_idx + i]
            pm = perform_df['pm'][start_idx + i]
            bw = perform_df['bw'][start_idx + i]
        else:
            fom = 0
            gain = 0
            pm = 0
            bw = 0

        g['fom'] = fom
        g['gain'] = gain
        g['pm'] = pm
        g['bw'] = bw


def add_topology_position(g):
    """Adds topological ordering positions to graph vertices.
    
    Performs topological sorting on a directed graph and assigns the resulting
    position index to each vertex as the 'topo' attribute.
    
    Args:
        g: igraph.Graph object (must be a directed acyclic graph).
        
    Side Effects:
        Modifies graph in-place by setting g.vs['topo'] to topological order indices.
    """
    # g is an igraph object
    topo_oreder = g.topological_sorting()
    num_nodes = len(topo_oreder)
    g.vs['topo'] = topo_oreder
    

def print_graph(g):
    """Prints graph structure and all vertex attributes.
    
    Args:
        g: igraph.Graph object to print.
        
    Side Effects:
        Prints graph summary and each vertex's attributes to stdout.
    """
    # logger.info(g)
    print(g)

    for i in range(g.vcount()):
        # logger.info(g.vs[i].attributes())
        print(g.vs[i].attributes())


#####################################################################################
#                           prepare the data for the model                          #
#####################################################################################
def get_contrastive_mask(batch_graphs):
    """Creates a contrastive learning mask for batch of graphs.
    
    Generates a mask that marks pairs of graphs with identical specifications
    (gain, bw, pm) as -inf, preventing them from being treated as negative pairs
    in contrastive learning.
    
    Args:
        batch_graphs: List of circuit graphs with 'gain', 'bw', 'pm' attributes.
        
    Returns:
        torch.Tensor: Shape (bsz, bsz) mask where -inf indicates same specification,
                     0 indicates different specifications.
                     
    Notes:
        Used in NCE (Noise Contrastive Estimation) loss to filter out false negatives.
    """
    bsz = len(batch_graphs)
    mask = torch.zeros(bsz, bsz)

    for i, gi in enumerate(batch_graphs):
        for j, gj in enumerate(batch_graphs):
            if j <= i:
                continue
            if (get_condition_mappings(gi['gain'], gi['bw'], gi['pm']) == get_condition_mappings(gj['gain'], gj['bw'], gj['pm'])):
                # print('i: ', gi['gain'], ' ', gi['bw'], ' ', gi['pm'])
                # print('j: ', gj['gain'], ' ', gj['bw'], ' ', gj['pm'])
                mask[i][j] = -torch.inf
                mask[j][i] = -torch.inf
                
                # print(mask[i][j])
    return mask


def floor_to_decimal(num, decimals=0):
    """Floors a number to a specified number of decimal places.
    
    Args:
        num: Number to floor.
        decimals: Number of decimal places to keep. Default 0 (integer floor).
        
    Returns:
        float: Floored number with specified decimal precision.
        
    Examples:
        >>> floor_to_decimal(3.789, decimals=1)
        3.7
        >>> floor_to_decimal(3.789, decimals=0)
        3
    """
    if decimals == 0:
        return math.floor(num)

    factor = 10 ** decimals
    return math.floor(num * factor) / factor


def count_conditions(dataset):
    """Counts the number of circuits for each unique specification combination.
    
    Creates a histogram of specification combinations in the dataset by encoding
    (gain, bw, pm) tuples into unique integer keys.
    
    Args:
        dataset: List of circuit graphs with 'gain', 'bw', 'pm' attributes.
        
    Returns:
        dict: Mapping from specification key (str) to count (int).
              Key format: gain*1 + bw*1000 + pm*1000000
              
    Notes:
        Specifications are floored to integers before encoding.
    """
    cond_keys = {'gain', 'bw', 'pm'}
    cond_combi_dict = {}
    cond_mp = {'gain': 1, 'bw': 1000, 'pm': 1000000}

    for i, g in enumerate(dataset):
        mp_key = 0
        for key in cond_keys:
            trunc_cond = floor_to_decimal(g[key], decimals=0)
            mp_key += trunc_cond * cond_mp[key]

        if str(mp_key) not in cond_combi_dict:
            cond_combi_dict[str(mp_key)] = 1
        else:
            cond_combi_dict[str(mp_key)] += 1
            
    return cond_combi_dict 


def add_start_symbol(g_origin):
    """Adds a special start symbol node to a circuit graph.
    
    Creates a new graph with an additional start node (type=2, path=2) at position 0,
    connected to the first node. Used for PACE architecture which requires explicit
    start symbols for autoregressive generation.
    
    Args:
        g_origin: Original igraph.Graph circuit with performance attributes.
        
    Returns:
        igraph.Graph: New graph with start symbol prepended, all original nodes
                     shifted by 1, and topological positions updated.
                     
    Notes:
        - Start node has type=2, path=2, r=c=gm=0
        - Original node types/paths are shifted: 0,1 stay same, others +1
        - Preserves performance attributes (gain, bw, pm, fom)
    """
    g = igraph.Graph(directed=True)
    g.add_vertices(g_origin.vcount() + 1)

    g['gain'] = g_origin['gain']
    g['bw'] = g_origin['bw']
    g['pm'] = g_origin['pm']
    g['fom'] = g_origin['fom']

    g.vs[0]['type'] = 2
    g.vs[0]['path'] = 2
    g.vs[0]['r'] = 0
    g.vs[0]['c'] = 0
    g.vs[0]['gm'] = 0

    g.add_edge(0, 1)

    for j in range(g_origin.vcount()):
        # add device parameters
        g.vs[j+1]['r'] = g_origin.vs[j]['r']  # r value in subg
        g.vs[j+1]['c'] = g_origin.vs[j]['c']  # c value in subg
        g.vs[j+1]['gm'] = g_origin.vs[j]['gm']  # gm value in subg

        # add node class for subgraph
        if g_origin.vs[j]['type'] == 0 or g_origin.vs[j]['type'] == 1:
            g.vs[j+1]['type'] = g_origin.vs[j]['type']
            g.vs[j+1]['path'] = g_origin.vs[j]['pos']
        else:
            g.vs[j+1]['type'] = g_origin.vs[j]['type']+1
            g.vs[j+1]['path'] = g_origin.vs[j]['pos']+1

        dest_lst = [g_origin.es[edge_index].tuple for edge_index in g_origin.incident(j)]
        for k in range(len(dest_lst)):
            g.add_edge(j+1, dest_lst[k][1]+1)

    add_topology_position(g)

    return g


def remove_start_symbol(g):
    """Removes the start symbol node from a PACE-format circuit graph.
    
    Deletes the first vertex (start symbol) and adjusts all node attributes
    (topo, path, type) by decrementing their values appropriately. Inverse
    operation of add_start_symbol().
    
    Args:
        g: igraph.Graph with start symbol at vertex 0.
        
    Side Effects:
        Modifies graph in-place by:
        - Deleting vertex 0
        - Decrementing all 'topo' values by 1
        - Decrementing 'path' values by 1 (except for special values 0, 1)
        - Decrementing 'type' values by 1 (except for special values 0, 1)
    """
    if g.vcount() == 0:
        return

    g.delete_vertices(0) # delete start symbol in pace architecture

    for i in range(g.vcount()):
        # add node class for subgraph
        g.vs[i]['topo'] = g.vs[i]['topo']-1

        if g.vs[i]['path'] != 0 and g.vs[i]['path'] != 1:
            g.vs[i]['path'] = g.vs[i]['path']-1

        if g.vs[i]['type'] != 0 and g.vs[i]['type'] != 1:
            g.vs[i]['type'] = g.vs[i]['type']-1


def do_clean(g_ori):
    """Cleans a circuit graph by removing unnecessary attributes and renaming 'pos' to 'path'.
    
    Creates a copy of the input graph, removes subgraph-related attributes that are
    not needed for model training, and adds topological position information.
    
    Args:
        g_ori: Original igraph.Graph with raw attributes from dataset.
        
    Returns:
        igraph.Graph: Cleaned graph with only essential attributes:
                     'type', 'path', 'r', 'c', 'gm', 'topo'.
                     
    Notes:
        Removed attributes: 'subg_ntypes', 'subg_nfeats', 'subg_adj', 'vid', 'pos'
        Renamed: 'pos' -> 'path'
    """
    g_cln = g_ori.copy()
    g_cln.vs['path'] = g_cln.vs['pos']
    
    del g_cln.vs['subg_ntypes']
    del g_cln.vs['subg_nfeats']
    del g_cln.vs['subg_adj']
    del g_cln.vs['vid']
    del g_cln.vs['pos']

    
    _ = add_topology_position(g_cln)

    return g_cln


def clean_datasets(args, train_data, test_data):
    """Cleans and filters training and test datasets based on model requirements.
    
    Performs data cleaning, filtering (removes negative PM, out-of-domain specs),
    and format conversion (adds start symbol for PACE, or cleans attributes).
    
    Args:
        args: Configuration dictionary with:
            - 'modeltype': Model type ('cktgen', 'cvaegan', 'ldt', 'evaluator', etc.)
            - 'archiname': Architecture name ('pace' or others)
        train_data: List of raw training circuit graphs.
        test_data: List of raw test circuit graphs.
        
    Returns:
        tuple: (train_data_cleaned, test_data_cleaned)
            - train_data_cleaned: Filtered and formatted training graphs
            - test_data_cleaned: Filtered and formatted test graphs
            
    Notes:
        Filtering rules for conditional models (cktgen, cvaegan, ldt, evaluator):
        - Removes circuits with negative phase margin (pm < 0)
        - Test set: removes specifications not present in training set
        
        Format conversion:
        - PACE architecture: adds start symbol via add_start_symbol()
        - Other architectures: cleans attributes via do_clean()
    """
    train_data_cleaned = []
    test_data_cleaned = []

    cond_keys = {'gain', 'bw', 'pm'}
    cond_mp = {'gain': 1, 'bw': 1000, 'pm': 1000000}

    train_mp = count_conditions(train_data)
    test_mp = count_conditions(test_data)

    for i, g in enumerate(train_data):

        # if ('conditioned' in args and args['conditioned']) or args['modeltype'] == 'predictor':
        if args['modeltype'] == 'cktgen' or args['modeltype'] == 'cvaegan' or args['modeltype'] == 'ldt' or args['modeltype'] == 'evaluator':

            if g['pm'] < 0:
                continue
            cond = 0

            for key in cond_keys:   
                trunc_cond = floor_to_decimal(g[key], decimals=0)
                cond += trunc_cond * cond_mp[key]

            # if train_mp[str(cond)] == 1:
            #     continue

        if args['archiname'] == 'pace':
            g_new = add_start_symbol(g)
        else:
            g_new = do_clean(g)

        train_data_cleaned.append(g_new)

    for i, g in enumerate(test_data):
        # if ('conditioned' in args and args['conditioned']) or args['modeltype'] == 'predictor':
        if args['modeltype'] == 'cktgen' or args['modeltype'] == 'cvaegan' or args['modeltype'] == 'ldt' or args['modeltype'] == 'evaluator':
            if g['pm'] < 0:
                continue
            cond = 0
            
            for key in cond_keys:
                trunc_cond = floor_to_decimal(g[key], decimals=0)
                cond += trunc_cond * cond_mp[key]

            if str(cond) not in train_mp: # delete num=1 and out of train domain data
                continue

        if args['archiname'] == 'pace':
            g_new = add_start_symbol(g)
        else:
            g_new = do_clean(g)
  
        test_data_cleaned.append(g_new)

    return train_data_cleaned, test_data_cleaned


def get_specifications(dataset):
    """Extracts unique specification combinations from a dataset.
    
    Args:
        dataset: List of circuit graphs with 'gain', 'bw', 'pm' attributes.
        
    Returns:
        list: List of unique [gain, bw, pm] specification tuples.
        
    Notes:
        Specifications are floored to integers before deduplication.
        Uses encoding: gain*1 + bw*1000 + pm*1000000 for uniqueness check.
    """
    spec_keys = {'gain', 'bw', 'pm'}
    spec_mp = {'gain': 1, 'bw': 1000, 'pm': 1000000}

    spec_dict = {}
    spec_idx = {}
    specs = []

    for i, g in enumerate(dataset):
        mp_key = 0
        for key in spec_keys:
            trunc_spec = floor_to_decimal(g[key], decimals=0)
            mp_key += trunc_spec * spec_mp[key]

        if str(mp_key) in spec_dict:
            continue
        else:
            spec_dict[str(mp_key)] = True
            specs.append([g['gain'], g['bw'], g['pm']])
            
    return specs


def get_specification_domain(dataset):
    """Clusters circuits by their specification domains (gain, bw, pm).
    
    Groups circuits with identical floored specifications into clusters.
    
    Args:
        dataset: List of circuit graphs with 'gain', 'bw', 'pm' attributes.
        
    Returns:
        list: List of clusters, where each cluster is a list of circuits
             sharing the same specification.
             
    Notes:
        Specifications are floored to integers before clustering.
        Uses encoding: gain*1 + bw*1000 + pm*1000000 for grouping.
    """
    spec_keys = {'gain', 'bw', 'pm'}
    spec_mp = {'gain': 1, 'bw': 1000, 'pm': 1000000}

    spec_cluster_ckt = []
    spec_dict = {}
    spec_idx = {}

    for i, g in enumerate(dataset):
        mp_key = 0

        for key in spec_keys:
            trunc_spec = floor_to_decimal(g[key], decimals=0)
            mp_key += trunc_spec * spec_mp[key]

        if str(mp_key) in spec_dict:
            idx = spec_idx[str(mp_key)]
            spec_cluster_ckt[idx].append(g)
        else:
            spec_dict[str(mp_key)] = True

            val = mp_key
            pm = floor_to_decimal(val / 1000000, decimals=0)
            val = val % 1000000
            bw = floor_to_decimal(val / 1000, decimals=0)
            val = val % 1000
            gain = val

            spec_cluster_ckt.append([g])
            spec_idx[str(mp_key)] = len(spec_cluster_ckt) - 1
            
    return spec_cluster_ckt


def get_condition_mappings(gain, bw, pm):
    """Encodes a specification tuple (gain, bw, pm) into a unique integer key.
    
    Args:
        gain: Gain value.
        bw: Bandwidth value.
        pm: Phase margin value.
        
    Returns:
        int: Unique encoding = floor(gain) + floor(bw)*1000 + floor(pm)*1000000
        
    Examples:
        >>> get_condition_mappings(2.5, 15.3, 45.7)
        45015002  # 45*1000000 + 15*1000 + 2
    """
    mappings = 0
    mappings += floor_to_decimal(gain, decimals=0)
    mappings += floor_to_decimal(bw, decimals=0) * 1000
    mappings += floor_to_decimal(pm, decimals=0) * 1000000
    return mappings


def get_datas_nums_more_than_k(dataset, k=10):
    """Filters circuits to keep only those in specification domains with ≥k samples.
    
    Clusters circuits by specifications, then retains only clusters with at least
    k circuits. Assigns a 'spec_id' attribute to each retained circuit.
    
    Args:
        dataset: List of circuit graphs with 'gain', 'bw', 'pm' attributes.
        k: Minimum cluster size threshold. Default 10.
        
    Returns:
        tuple: (filtered_circuits, labels, num_specs)
            - filtered_circuits: List of circuits in large-enough clusters
            - labels: List of spec_id for each circuit
            - num_specs: Number of unique specification domains retained
            
    Side Effects:
        Adds 'spec_id' attribute to each circuit in the result.
    """
    spec_keys = {'gain', 'bw', 'pm'}
    spec_mp = {'gain': 1, 'bw': 1000, 'pm': 1000000}

    spec_cluster_ckt = []
    spec_dict = {}
    spec_idx = {}

    for i, g in enumerate(dataset):
        mp_key = 0

        for key in spec_keys:
            trunc_spec = floor_to_decimal(g[key], decimals=0)
            mp_key += trunc_spec * spec_mp[key]

        if str(mp_key) in spec_dict:
            idx = spec_idx[str(mp_key)]
            spec_cluster_ckt[idx].append(g)
        else:
            spec_dict[str(mp_key)] = True

            val = mp_key
            pm = floor_to_decimal(val / 1000000, decimals=0)
            val = val % 1000000
            bw = floor_to_decimal(val / 1000, decimals=0)
            val = val % 1000
            gain = val

            spec_cluster_ckt.append([g])
            spec_idx[str(mp_key)] = len(spec_cluster_ckt) - 1
    
    res = []
    labels = []
    colors = []
    spec_id = 0
    for i, cluster in enumerate(spec_cluster_ckt):
        if len(cluster) >= k:
            for _, g in enumerate(cluster):
                g['spec_id'] = spec_id
                
            res = res + cluster
            labels= labels + [spec_id] * len(cluster)
            spec_id += 1
    print(len(res))
    return res, labels, spec_id


def get_specification(args, datasets):
    """Extracts continuous specification values from training dataset.
    
    Args:
        args: Configuration dictionary with 'device' key.
        datasets: Dictionary with 'train' key containing circuit graphs.
        
    Returns:
        dict: Dictionary with keys 'continuous_gains', 'continuous_bws',
             'continuous_pms', 'continuous_foms' as torch tensors.
             
    Note:
        This function appears to have a bug - it references 'batch_graphs'
        which is not defined in the parameter list.
    """
    continuous_gains  = torch.tensor([g['gain'] for g in datasets['train']], dtype=torch.float32).to(args['device'])
    continuous_bws    = torch.tensor([g['bw'] for g in batch_graphs], dtype=torch.float32).to(args['device'])
    continuous_pms    = torch.tensor([g['pm'] for g in batch_graphs], dtype=torch.float32).to(args['device'])
    continuous_foms   = torch.tensor([g['fom'] for g in batch_graphs], dtype=torch.float32).to(args['device'])

    return {
        'continuous_gains': continuous_gains, 'continuous_bws': continuous_bws, 
        'continuous_pms': continuous_pms, 'continuous_foms': continuous_foms
    }


def get_one_specifications(args, g):
    """Extracts and floors specifications from a single circuit graph.
    
    Args:
        args: Configuration dictionary (currently unused).
        g: Circuit graph with 'gain', 'bw', 'pm' attributes.
        
    Returns:
        dict: Dictionary with floored integer values for 'gain', 'bw', 'pm'.
    """
    gain = floor_to_decimal(g['gain'], decimals=0)
    bw = floor_to_decimal(g['bw'], decimals=0)
    pm = floor_to_decimal(g['pm'], decimals=0)

    return {
        'gain': gain, 
        'bw': bw, 
        'pm': pm
    }


def get_fom_train_mean_and_std(args, train_graphs):
    """Computes mean and standard deviation of FoM values in training set.
    
    Args:
        args: Configuration dictionary with 'device' key.
        train_graphs: List of training circuit graphs with 'fom' attributes.
        
    Returns:
        tuple: (mean, std) tensors for FoM normalization.
    """
    cont_foms   = torch.tensor([g['fom'] for g in train_graphs], dtype=torch.float).to(args['device'])
    mean, std = cont_foms.mean(), cont_foms.std()

    return mean, std

def standard_fom(mean, std, fom):
    """Standardizes FoM values using z-score normalization.
    
    Args:
        mean: Mean value for normalization.
        std: Standard deviation for normalization.
        fom: FoM value(s) to standardize (scalar or tensor).
        
    Returns:
        Standardized FoM: (fom - mean) / std
    """
    return (fom - mean) / std


#####################################################################################
#                           prepare the data for the model                          #
#####################################################################################
def transform_sizes(max_n, batch_graphs):
    """Extracts device sizing parameters (r, c, gm) organized by path position.
    
    For each graph, creates a flat vector of size 3*max_n where positions
    [path*3, path*3+1, path*3+2] contain [r, c, gm] values for that path.
    
    Args:
        max_n: Maximum number of paths (nodes) in circuit.
        batch_graphs: List of circuit graphs with vertex attributes 'path', 'r', 'c', 'gm'.
        
    Returns:
        torch.FloatTensor: Shape (batch_size, 3*max_n) with device sizing parameters.
        
    Side Effects:
        - Adds topological positions to graphs
        - Clips path values > max_n-1 to max_n-1
    """
    v_sizes = []
    for g in batch_graphs:
        _ = add_topology_position(g)
        v_size = [0] * (3 * max_n)
        for v_ in range(len(g.vs)):
            if g.vs[v_]["path"] > max_n - 1:
                g.vs[v_]["path"] = max_n - 1

            _path_ = g.vs[v_]["path"]
            v_size[_path_ * 3 + 0] = g.vs[v_]["r"]   # r value in subg
            v_size[_path_ * 3 + 1] = g.vs[v_]["c"]   # c value in subg
            v_size[_path_ * 3 + 2] = g.vs[v_]["gm"]  # gm value in subg
        v_sizes.append(v_size)

    v_sizes = torch.FloatTensor(v_sizes)
    return v_sizes


def transform_digin(args, batch_graphs):
    """Transforms batch of graphs for DIGIN architecture.
    
    Args:
        args: Configuration dictionary with 'max_n' and 'archiname' keys.
        batch_graphs: List of circuit graphs.
        
    Returns:
        dict: Batch dictionary with 'G' (graphs) and 'v_sizes' (sizing parameters).
    """
    max_n = args['max_n']
    if args['archiname']=='pace':
        max_n -= 1
    batch = {'G': batch_graphs}
    v_sizes = transform_sizes(max_n, batch_graphs)
    batch['v_sizes'] = v_sizes
    return batch


def transform_cktarchi(args, batch_graphs):
    """Transforms batch of graphs into format for CktArchi-based models.
    
    Prepares comprehensive batch data including node types, paths, topological orders,
    adjacency matrices, and device sizing parameters. Used by CktGNN, CVAEGAN, etc.
    
    Args:
        args: Configuration dictionary with 'max_n', 'archiname', and 'device' keys.
        batch_graphs: List of circuit graphs.
        
    Returns:
        dict: Batch dictionary with keys:
            - 'G': Original graphs
            - 'v_sizes': Device sizing parameters (bsz, 3*max_n)
            - 'v_types': Node types (bsz, max_n), padded with 0
            - 'v_paths': Path positions (bsz, max_n), padded with 0
            - 'gnd_types': Ground truth types (bsz, max_n), padded with -1
            - 'gnd_paths': Ground truth paths (bsz, max_n), padded with -1
            - 'v_topos_1hot': One-hot topological positions (bsz, max_n, max_n)
            - 'adj': Adjacency matrices (bsz, max_n, max_n)
            - 'num_nodes': List of actual node counts per graph
            - 'gnd_edges': Ground truth edge vectors for DAG reconstruction
            
    Notes:
        For PACE architecture, max_n is decremented by 1 (to exclude start symbol).
    """
    batch = {'G': batch_graphs}
    bsz = len(batch_graphs)
    max_n = args['max_n']

    if args['archiname']=='pace':
        max_n -= 1
        
    v_sizes = transform_sizes(max_n, batch_graphs)

    batch['v_sizes'] = v_sizes
    batch['v_types'] = torch.tensor([seq_padding(ori_lst=g.vs['type'], 
                                                 max_len=max_n, 
                                                 pad_val=0) 
                                                 for g in batch_graphs], dtype=torch.int64)                                      
    batch['v_paths'] = torch.tensor([seq_padding(ori_lst=g.vs['path'], # not continuous
                                                 max_len=max_n,
                                                 pad_val=0)
                                                 for g in batch_graphs], dtype=torch.int64)


    batch['gnd_types'] = torch.tensor([seq_padding(ori_lst=g.vs['type'], 
                                                 max_len=max_n,
                                                 pad_val=-1) 
                                                 for g in batch_graphs], dtype=torch.int64)                                      
    batch['gnd_paths'] = torch.tensor([seq_padding(ori_lst=g.vs['path'], # not continuous
                                                 max_len=max_n,
                                                 pad_val=-1)
                                                 for g in batch_graphs], dtype=torch.int64)


    v_topos_lst = [g.vs['topo'] for g in batch_graphs]  # padding with self.max_n - 1
    batch['v_topos_1hot'] = get_one_hot(ori_lst=v_topos_lst, 
                                        num_cls=max_n, 
                                        max_len=max_n, 
                                        pad_val=max_n - 1)  # max(node_position) + 1
    

    ##### ---- init mask ---- #####
    num_nodes = []
    gnd_edges = []

    adj = torch.zeros(bsz, max_n, max_n)

    ##### ---- get features for each graph ---- #####
    for i in range(bsz):
        g = batch_graphs[i]
        num_node = g.vcount()
        num_nodes.append(num_node)
        adj_i = torch.FloatTensor(g.get_adjacency().data)
        adj[i, :g.vcount(), :g.vcount()] = adj_i
        cnt = 0
        
        num_pot_edges = int(num_node * (num_node - 1) / 2.0) # max edges for a dag
        gnd_edge = torch.zeros(num_pot_edges, 1)
        for v in range(num_node-1, 0, -1): 
            gnd_edge[cnt:cnt+v, :] = adj[i, :v, v].view(v, 1)
            cnt += v
        gnd_edges.append(gnd_edge)
    

    batch['adj'], batch['num_nodes'] = adj, num_nodes
    batch['gnd_edges'] = gnd_edges

    batch = {key: val.to(args['device']) if torch.is_tensor(val) else val 
                                         for key, val in batch.items()}
                                         
    return batch


def transform_pace(args, batch_graphs):
    """Transforms batch of graphs for PACE architecture.
    
    Args:
        args: Configuration dictionary (unused).
        batch_graphs: List of circuit graphs.
        
    Returns:
        dict: Minimal batch dictionary with only 'G' (graphs).
        
    Notes:
        PACE handles graph processing internally, so no preprocessing needed.
    """
    return {'G': batch_graphs}


def transform_cktgnn(args, batch_graphs):
    """Transforms batch of graphs for CktGNN architecture.
    
    Args:
        args: Configuration dictionary (unused).
        batch_graphs: List of circuit graphs.
        
    Returns:
        dict: Minimal batch dictionary with only 'G' (graphs).
        
    Notes:
        CktGNN handles graph processing internally via DGL, so no preprocessing needed.
    """
    return {'G': batch_graphs}


def transform_specification(args, batch_graphs):
    """Extracts and transforms specification values from circuit graphs.
    
    Floors gain, bw, pm to integers and keeps fom as float.
    
    Args:
        args: Configuration dictionary with 'device' key.
        batch_graphs: List of circuit graphs with 'gain', 'bw', 'pm', 'fom' attributes.
        
    Returns:
        dict: Dictionary with torch tensors:
            - 'gains': Int64 tensor of floored gain values
            - 'bws': Int64 tensor of floored bandwidth values
            - 'pms': Int64 tensor of floored phase margin values
            - 'foms': Float tensor of figure-of-merit values
    """
    # truncate
    gains  = torch.tensor([floor_to_decimal(g['gain'], decimals=0) for g in batch_graphs], dtype=torch.int64).to(args['device'])
    bws    = torch.tensor([floor_to_decimal(g['bw'], decimals=0) for g in batch_graphs], dtype=torch.int64).to(args['device'])
    pms    = torch.tensor([floor_to_decimal(g['pm'], decimals=0) for g in batch_graphs], dtype=torch.int64).to(args['device'])
    foms   = torch.tensor([g['fom'] for g in batch_graphs], dtype=torch.float).to(args['device'])

    return {'gains': gains, 'bws': bws, 'pms': pms, 'foms': foms}


def transform_specification_directly(args, gains, bws, pms):
    """Transforms specification values directly from arrays (not from graphs).
    
    Args:
        args: Configuration dictionary with 'device' key.
        gains: List or array of gain values.
        bws: List or array of bandwidth values.
        pms: List or array of phase margin values.
        
    Returns:
        dict: Dictionary with int64 torch tensors for 'gains', 'bws', 'pms',
             each floored to integers.
    """
    # truncate
    gains  = torch.tensor([floor_to_decimal(gain, decimals=0) for gain in gains], dtype=torch.int64).to(args['device'])
    bws    = torch.tensor([floor_to_decimal(bw, decimals=0) for bw in bws], dtype=torch.int64).to(args['device'])
    pms    = torch.tensor([floor_to_decimal(pm, decimals=0) for pm in pms], dtype=torch.int64).to(args['device'])

    return {'gains': gains, 'bws': bws, 'pms': pms}


def transforms(args, batch_graphs, mode='train'):
    """Main transformation function that prepares graph batches for model input.
    
    Dynamically selects the appropriate transformation function based on architecture,
    then adds specifications and contrastive masks as needed.
    
    Args:
        args: Configuration dictionary with keys:
            - 'archiname': Architecture name ('pace', 'cktgnn', 'cktarchi', 'digin')
            - 'device': Target device for tensors
        batch_graphs: List of circuit graphs to transform.
        mode: Operating mode, 'train' or 'generate'. Default 'train'.
        
    Returns:
        dict: Transformed batch dictionary with architecture-specific format,
             plus specifications and contrastive mask (unless mode='generate').
             All tensors are moved to args['device'].
             
    Notes:
        - Calls transform_{archiname}() dynamically
        - Adds specifications via transform_specification() if not in generate mode
        - Always adds contrastive learning filter mask
        - Moves all tensor values to GPU/device
    """
    transform = getattr(utils_data, f"transform_{args['archiname']}")
    batch = transform(args, batch_graphs)

    # if args['conditioned'] and mode != 'generate':
    if mode != 'generate':
        batch.update(transform_specification(args, batch_graphs))
        
    # if args['contrastive']:
        batch['filter_mask'] = get_contrastive_mask(batch_graphs)
            
    # if args['modeltype'] == 'evaluator':
    #     batch.update(transform_continues_specification(args, batch_graphs))

    # put tensor to gpu if use cuda
    batch = {key: val.to(args['device']) if torch.is_tensor(val) else val 
                                         for key, val in batch.items()}
    return batch
"""Dataset loaders for circuit benchmarks.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import pickle
import argparse
import utils.data as utils_data
import utils.paths as utils_paths
import pandas as pd

def get_datasets(args):
    """Loads and prepares Op-amp Circuit Benchmark (OCB) datasets.
    
    Loads circuit graphs from pickle files, adds performance specifications
    from CSV files, and performs data cleaning based on model requirements.
    
    Supported benchmarks:
    - CktBench101: 101 unique circuit topologies (9,000 train + test samples)
    - CktBench301: 301 unique circuit topologies (40,000 train + test samples)
    
    Args:
        args: Configuration dictionary with keys:
            - 'data_fold_name': Dataset folder ('CktBench101' or 'CktBench301')
            - 'data_name': Base filename ('ckt_bench_101' or 'ckt_bench_301')
            - 'archiname': Architecture name (affects max_n, START_SYMBOL)
            
    Returns:
        dict: Dataset dictionary with keys:
            - 'train': List of training circuit graphs (igraph.Graph objects)
            - 'test': List of test circuit graphs (igraph.Graph objects)
            
    Side Effects:
        Modifies args dictionary by adding:
        - 'file_dir', 'data_dir', 'data_type': Path configurations
        - 'START_TYPE', 'END_TYPE': Node type indices (0, 1)
        - 'num_gain_type', 'num_bw_type', 'num_pm_type': Discretization bins
        - 'max_n': Max nodes (8 for standard, 9 for PACE)
        - 'START_SYMBOL': Start symbol type (only for PACE)
        
    Notes:
        Each circuit graph has attributes:
        - 'type', 'path', 'topo': Node attributes
        - 'r', 'c', 'gm': Device parameters
        - 'gain', 'bw', 'pm', 'fom': Performance specifications (added from CSV)
    """
    
    ##### ---- read from path ---- #####
    datasets = {}
    
    # Use centralized path management for reliable path resolution
    args['data_type'] = 'igraph'
    
    pkl_name = os.path.join(args['data_dir'], args['data_name'] + '_' + args['data_type'] + '.pkl')

    print('data path: ', pkl_name)

    with open(pkl_name, 'rb') as f:
        all_datasets = pickle.load(f)

    ##### ---- data configs ---- #####
    args['START_TYPE'] = 0
    args['END_TYPE'] = 1

    if args['data_name'] == 'ckt_bench_301':
        train_start = 0
        test_start = 40000
    
        train_datasets = [all_datasets[i][0] for i in range(train_start, test_start)]
        test_datasets = [all_datasets[i][0] for i in range(test_start, len(all_datasets))]
    else:
        train_start = 0
        test_start = 9000
        
        train_datasets = all_datasets[0]
        test_datasets = all_datasets[1]

        train_datasets = [train_datasets[i][0] for i in range(len(train_datasets))]
        test_datasets = [test_datasets[i][0] for i in range(len(test_datasets))]

    ##### ---- read conditions ---- #####
    if args['data_name'] == 'ckt_bench_301':
        performance_name = str(utils_paths.get_performance_file(args['data_fold_name'], '301'))
        ##### ---- keep single data ---- #####
        args['num_gain_type'] = 4
        args['num_bw_type'] = 19
        args['num_pm_type'] = 5
        
        
        # ##### ---- keep single data ---- #####
        # args['num_gain_type'] = 19
        # args['num_pm_type'] = 5
        # args['num_bw_type'] = 4
    else:
        performance_name = str(utils_paths.get_performance_file(args['data_fold_name'], '101'))
        ##### ---- keep single data point ---- #####
        args['num_gain_type'] = 4
        args['num_bw_type'] = 32
        args['num_pm_type'] = 6
         

        ##### ---- remove single data points ---- #####
        # args['num_gain_type'] = 4
        # args['num_pm_type'] = 5 
        # args['num_bw_type'] = 32
        
    performance_dataframe = pd.read_csv(performance_name)
    utils_data.add_conditions(train_datasets, 
                              performance_dataframe, 
                              start_idx=train_start)

    utils_data.add_conditions(test_datasets, 
                              performance_dataframe, 
                              start_idx=test_start)
    
    ##### ---- no start symbol ---- #####
    if args['archiname'] == 'pace':
        args['max_n'] = 9
        args['START_SYMBOL'] = 2
    else:
        args['max_n'] = 8

    train_datasets, test_datasets = utils_data.clean_datasets(args, train_datasets, test_datasets)
    return {'train': train_datasets, 'test': test_datasets}
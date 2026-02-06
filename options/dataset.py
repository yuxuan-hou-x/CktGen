"""Dataset-specific configuration options.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import argparse

def add_dataset_options(parser):
    """
    Add dataset configuration options.
    These options specify which circuit benchmark dataset to use for training and evaluation.
    The codebase supports CktBench101 (101 circuits) and CktBench301 (301 circuits).
    """
    group = parser.add_argument_group(description='Dataset options')

    group.add_argument(
        '--data_fold_name', 
        default='CktBench101', 
        help='Dataset folder name under dataset/OCB/. Options: "CktBench101" (101 unique circuit topologies) '
             'or "CktBench301" (301 unique circuit topologies). This determines which benchmark to load.'
    )
    group.add_argument(
        '--data_type', 
        default='OCB', 
        help='Dataset type identifier. Currently supports "OCB" (Op-amp Circuit Benchmark). '
             'Used to locate the dataset under dataset/{data_type}/{data_fold_name}/'
    )
    group.add_argument(
        '--data_name', 
        default='ckt_bench_101', 
        help='Base name of the pickle file containing circuit graphs. The file will be loaded from '
             'dataset/{data_type}/{data_fold_name}/{data_name}_igraph.pkl. '
             'Use "ckt_bench_101" for CktBench101 or "ckt_bench_301" for CktBench301.'
    )
    group.add_argument(
        '--graph_numbers', 
        type=int, 
        default=10000, 
        help='Number of circuits to process for statistical analysis and visualization tasks. '
             'Used in scripts/visualization/ scripts (statistic.sh, infer_time.sh) to control how many '
             'circuits are analyzed when computing statistics or measuring inference time. '
             'Not used during training - training always uses the full dataset. '
             'Typical values: 10000 for quick analysis, 50000 for comprehensive statistics. '
             'Note: This is an upper limit; if the dataset has fewer circuits, all circuits will be used.'
    )
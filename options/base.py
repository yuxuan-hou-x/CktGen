"""Base configuration options.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


from argparse import ArgumentParser  # noqa

def add_misc_options(parser):
    """
    Add miscellaneous experiment configuration options.
    These options control the output directory structure and experiment naming.
    """
    group = parser.add_argument_group(description='Miscellaneous options')
    group.add_argument(
        '--out_dir', 
        type=str, 
        default='../output', 
        help='Root directory for saving all experimental outputs including checkpoints, logs, and generated circuits. '
             'A subdirectory named by --exp_name will be created inside this directory.'
    )
    group.add_argument(
        '--exp_name', 
        type=str, 
        default='exp_debug', 
        help='Experiment name used to create a unique subdirectory inside --out_dir. '
             'All outputs for this experiment (model checkpoints, training logs, evaluation results) '
             'will be saved to {out_dir}/{exp_name}/. '
    )
    group.add_argument(
        '--print_iter', 
        type=int, 
        default=20, 
        help='Frequency (in batches) for printing training progress to logger. '
             'Every print_iter batches, detailed loss metrics will be logged (if logging code is enabled). '
             'Note: Currently the detailed logging code is commented out in training scripts '
             '(train_cktgen.py, train_ldt.py, train_evaluator.py), but this parameter is preserved for '
             'future debugging and development. Default 20 batches.'
    )

def add_cuda_options(parser):
    """
    Add CUDA device configuration options.
    Controls whether to use GPU or CPU for training and inference.
    """
    group = parser.add_argument_group('Cuda options')
    group.add_argument(
        '--cuda', 
        dest='cuda', 
        action='store_true', 
        help='Enable GPU acceleration if CUDA is available. The code will automatically fall back to CPU if no GPU is detected.'
    )
    group.add_argument(
        '--cpu', 
        dest='cuda', 
        action='store_false', 
        help='Force CPU usage even if GPU is available. Useful for debugging or when GPU memory is limited.'
    )
    group.set_defaults(cuda=True)

def adding_cuda(args):
    """Configures CUDA device based on availability and user preferences.
    
    Sets the 'device' key in args to either CUDA or CPU torch.device based on
    whether CUDA is requested and available.
    
    Args:
        args: Configuration dictionary with 'cuda' key (bool).
              Modified in-place to add 'device' key.
              
    Side Effects:
        Modifies args dictionary by adding 'device' key with torch.device object.
    """
    import torch
    if args['cuda'] and torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
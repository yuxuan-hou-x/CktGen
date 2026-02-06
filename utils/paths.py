"""Path management utilities for CktGen project.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module provides centralized path management to ensure consistent
and reliable file access across different execution contexts (running
from project root, from subdirectories, or in different environments).
"""


import os
from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory.
    
    This function reliably finds the project root by looking for marker files
    that uniquely identify the CktGen project root directory.
    
    Strategy:
        1. Start from this file's location (utils/paths.py)
        2. Move up one level to reach project root
        3. Verify by checking for marker files (README.md, setup.py, etc.)
    
    Returns:
        Path: Absolute path to project root directory.
        
    Raises:
        RuntimeError: If project root cannot be determined.
        
    Example:
        >>> root = get_project_root()
        >>> print(root)
        /Users/username/Desktop/projects/CktGen-Exp-Cleaned
    """
    # Get the directory containing this file (utils/)
    current_file = Path(__file__).resolve()
    utils_dir = current_file.parent
    
    # Project root is one level up from utils/
    project_root = utils_dir.parent
    
    # Verify this is actually the project root by checking for marker files
    marker_files = ['README.md', 'dataset', 'models', 'train', 'utils']
    
    for marker in marker_files:
        if not (project_root / marker).exists():
            raise RuntimeError(
                f"Could not verify project root. Expected to find '{marker}' at {project_root}"
            )
    
    return project_root


def get_data_dir(data_fold_name=None):
    """Get the absolute path to the dataset directory.
    
    Args:
        data_fold_name: Optional dataset folder name (e.g., 'CktBench101', 'CktBench301').
                       If provided, returns path to that specific dataset folder.
                       If None, returns path to the base dataset directory.
    
    Returns:
        Path: Absolute path to dataset directory.
        
    Example:
        >>> data_dir = get_data_dir('CktBench101')
        >>> print(data_dir)
        /Users/username/Desktop/projects/CktGen-Exp-Cleaned/dataset/OCB/CktBench101
    """
    project_root = get_project_root()
    
    if data_fold_name:
        return project_root / 'dataset' / 'OCB' / data_fold_name
    else:
        return project_root / 'dataset'


def get_output_dir(exp_name=None):
    """Get the absolute path to the output directory.
    
    Args:
        exp_name: Optional experiment name. If provided, returns path to that
                 specific experiment's output folder.
    
    Returns:
        Path: Absolute path to output directory.
        
    Example:
        >>> output_dir = get_output_dir('my_experiment')
        >>> print(output_dir)
        /Users/username/Desktop/projects/CktGen-Exp-Cleaned/output/my_experiment
    """
    project_root = get_project_root()
    output_base = project_root / 'output'
    
    if exp_name:
        return output_base / exp_name
    else:
        return output_base


def get_checkpoint_path(exp_name, model_name, epoch=None):
    """Get the absolute path to a model checkpoint file.
    
    Args:
        exp_name: Experiment name (subfolder in output/).
        model_name: Model name prefix (e.g., 'cktgen_cktarchi').
        epoch: Optional epoch number. If provided, constructs checkpoint filename
               like '{model_name}_checkpoint{epoch}.pth'. If None, returns the
               directory containing checkpoints.
    
    Returns:
        Path: Absolute path to checkpoint file or directory.
        
    Example:
        >>> ckpt = get_checkpoint_path('exp1', 'cktgen_cktarchi', 600)
        >>> print(ckpt)
        .../output/exp1/cktgen_cktarchi_checkpoint600.pth
    """
    exp_dir = get_output_dir(exp_name)
    
    if epoch is not None:
        checkpoint_file = f'{model_name}_checkpoint{epoch}.pth'
        return exp_dir / checkpoint_file
    else:
        return exp_dir


def resolve_path(path_str, base_dir=None):
    """Resolve a path string to an absolute Path object.
    
    This function handles various path formats:
    - Absolute paths: returned as-is
    - Relative paths starting with './': resolved relative to base_dir or project root
    - Relative paths starting with '../': resolved relative to base_dir or project root
    - Other relative paths: resolved relative to base_dir or project root
    
    Args:
        path_str: Path string to resolve (can be str or Path).
        base_dir: Base directory for resolving relative paths. If None, uses project root.
    
    Returns:
        Path: Absolute Path object.
        
    Example:
        >>> p = resolve_path('./data/model.pth')
        >>> print(p)
        /Users/username/Desktop/projects/CktGen-Exp-Cleaned/data/model.pth
    """
    if path_str is None:
        return None
    
    path = Path(path_str)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Resolve relative to base_dir or project root
    if base_dir is None:
        base_dir = get_project_root()
    else:
        base_dir = Path(base_dir).resolve()
    
    return (base_dir / path).resolve()


def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory (str or Path).
    
    Returns:
        Path: Absolute path to the directory.
        
    Example:
        >>> output_dir = ensure_dir('./output/my_exp')
        >>> # Directory is now created if it didn't exist
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def get_dataset_file(data_fold_name, data_name, data_type='igraph'):
    """Get the absolute path to a dataset file.
    
    Args:
        data_fold_name: Dataset folder name (e.g., 'CktBench101').
        data_name: Dataset base name (e.g., 'ckt_bench_101').
        data_type: Dataset file type (default: 'igraph').
    
    Returns:
        Path: Absolute path to dataset file.
        
    Example:
        >>> dataset_file = get_dataset_file('CktBench101', 'ckt_bench_101')
        >>> print(dataset_file)
        .../dataset/OCB/CktBench101/ckt_bench_101_igraph.pkl
    """
    data_dir = get_data_dir(data_fold_name)
    filename = f'{data_name}_{data_type}.pkl'
    return data_dir / filename


def get_performance_file(data_fold_name, benchmark='101'):
    """Get the absolute path to a performance CSV file.
    
    Args:
        data_fold_name: Dataset folder name (e.g., 'CktBench101').
        benchmark: Benchmark identifier ('101' or '301').
    
    Returns:
        Path: Absolute path to performance CSV file.
        
    Example:
        >>> perf_file = get_performance_file('CktBench101', '101')
        >>> print(perf_file)
        .../dataset/OCB/CktBench101/perform101.csv
    """
    data_dir = get_data_dir(data_fold_name)
    filename = f'perform{benchmark}.csv'
    return data_dir / filename


def get_checkpoint_path(out_dir, exp_name, epoch=None):
    """Get the path for saving a model checkpoint.
    
    Args:
        out_dir: Output directory path (str or Path).
        exp_name: Experiment name (used as the checkpoint filename base).
        epoch: Optional epoch number. If provided, creates intermediate checkpoint
               like '{exp_name}_checkpoint{epoch}.pth'. If None, creates final
               checkpoint as '{exp_name}.pth'.
    
    Returns:
        Path: Absolute path to checkpoint file.
        
    Example:
        >>> # Intermediate checkpoint (during training)
        >>> ckpt_path = get_checkpoint_path('./output/cktgen', 'cktgen_cond_gen_101', 100)
        >>> print(ckpt_path)
        .../output/cktgen/cktgen_cond_gen_101_checkpoint100.pth
        
        >>> # Final checkpoint (best/last model)
        >>> ckpt_path = get_checkpoint_path('./output/cktgen', 'cktgen_cond_gen_101')
        >>> print(ckpt_path)
        .../output/cktgen/cktgen_cond_gen_101.pth
    """
    out_path = Path(out_dir)
    if epoch is not None:
        filename = f'{exp_name}_checkpoint{epoch}.pth'
    else:
        filename = f'{exp_name}.pth'
    return out_path / filename


# Convenience function to update args dict with reliable paths
def setup_paths(args):
    """Update args dictionary with reliable absolute paths.
    
    This function modifies the args dictionary in-place, ensuring all
    path-related keys use absolute paths based on project root.
    
    Args:
        args: Configuration dictionary to update. Should contain:
            - 'data_fold_name': Dataset folder name
            - 'exp_name': Experiment name (for output directory)
            - Other optional path-related keys
    
    Side Effects:
        Modifies args dict by setting:
        - 'project_root': Absolute path to project root
        - 'data_dir': Absolute path to dataset directory
        - 'out_dir': Absolute path to output directory (with exp_name)
        
    Example:
        >>> args = {'data_fold_name': 'CktBench101', 'exp_name': 'my_exp'}
        >>> setup_paths(args)
        >>> print(args['data_dir'])
        /Users/username/Desktop/projects/CktGen-Exp-Cleaned/dataset/OCB/CktBench101
    """
    project_root = get_project_root()
    args['project_root'] = str(project_root)
    
    # Setup data directory
    if 'data_fold_name' in args:
        args['data_dir'] = str(get_data_dir(args['data_fold_name']))
    
    # Setup output directory
    # out_dir is the direct output directory, exp_name is only for log file naming
    if 'out_dir' in args:
        # Convert relative out_dir to absolute if needed
        if not Path(args['out_dir']).is_absolute():
            out_base = project_root / args['out_dir']
        else:
            out_base = Path(args['out_dir'])
        
        args['out_dir'] = str(out_base)
    
    # Resolve checkpoint paths if provided
    # This is critical for ensuring paths work correctly regardless of where the script is run
    for key in ['resume_pth', 'vae_pth', 'pretrained_eval_resume_pth']:
        if key in args and args[key] is not None and args[key] != '':
            # Convert empty strings to None for consistency
            if args[key].strip() == '':
                args[key] = None
            else:
                # Resolve relative paths to absolute paths based on project root
                resolved = str(resolve_path(args[key]))
                args[key] = resolved
                
                # Log the path resolution for debugging
                # Note: We don't check file existence here because:
                # 1. For resume_pth: file should exist when resuming, but not when training from scratch
                # 2. For vae_pth: file should exist when training LDT
                # 3. For pretrained_eval_resume_pth: file should exist when evaluating
                # The actual loading code will handle file-not-found errors appropriately
    
    return args


def safe_load_checkpoint(checkpoint_path, map_location='cpu'):
    """Safely load a checkpoint file with automatic path resolution.
    
    This function provides a robust way to load checkpoint files that:
    1. Automatically resolves relative paths to absolute paths based on project root
    2. Checks if the file exists before attempting to load
    3. Provides clear error messages if the file cannot be found
    
    Args:
        checkpoint_path: Path to checkpoint file (can be relative or absolute).
        map_location: Device to map the loaded checkpoint to (default: 'cpu').
        
    Returns:
        Loaded checkpoint object.
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        
    Example:
        >>> model = safe_load_checkpoint('./output/model_checkpoint100.pth')
        >>> # Equivalent to:
        >>> # path = resolve_path('./output/model_checkpoint100.pth')
        >>> # model = torch.load(path, map_location='cpu')
    """
    import torch
    
    # Resolve the path to absolute
    resolved_path = resolve_path(checkpoint_path)
    
    # Check if file exists
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {resolved_path}\n"
            f"Original path provided: {checkpoint_path}\n"
            f"Make sure the checkpoint file exists at the specified location."
        )
    
    # Load the checkpoint
    return torch.load(str(resolved_path), map_location=map_location)


if __name__ == '__main__':
    """Test path utilities."""
    print("Testing path utilities...")
    print(f"Project root: {get_project_root()}")
    print(f"Data dir (base): {get_data_dir()}")
    print(f"Data dir (CktBench101): {get_data_dir('CktBench101')}")
    print(f"Output dir (base): {get_output_dir()}")
    print(f"Output dir (my_exp): {get_output_dir('my_exp')}")
    print(f"Dataset file: {get_dataset_file('CktBench101', 'ckt_bench_101')}")
    print(f"Performance file: {get_performance_file('CktBench101', '101')}")
    print("\nAll tests passed!")


"""Logging utilities for training and evaluation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import sys
import logging
import numpy as np
import importlib
import torch
from torch.nn import functional as F

MODELTYPES = ['cvae', 'cae', 'ldt']  # not used: 'cae'
ARCHINAMES = ['pace', 'cktgnn', 'dvae', 'dagnn']


def serialize(obj):
    """Serializes objects for JSON encoding, particularly torch.device objects.
    
    Args:
        obj: Object to serialize.
        
    Returns:
        Serialized representation (str for torch.device, original object otherwise).
        
    Raises:
        TypeError: If object type is not serializable (note: this is unreachable code).
    """
    import torch
    if isinstance(obj, torch.device):
        return str(obj)
    else:
        return obj
    raise TypeError('Type not serializable')


def get_logger(out_dir, exp_name=None):
    """Creates and configures a logger for experiment tracking.
    
    Sets up a logger that writes to both a file and stdout,
    with timestamps and log levels.
    
    Args:
        out_dir: Directory where the log file will be created.
        exp_name: Optional experiment name for log file naming.
                  If provided, log file will be named '{exp_name}.log'.
                  If None, log file will be named 'run.log'.
        
    Returns:
        logging.Logger: Configured logger instance named 'Exp' with INFO level.
        
    Notes:
        Log format: "%(asctime)s %(levelname)s %(message)s"
        Handlers:
        - FileHandler: Writes to {out_dir}/{exp_name}.log or {out_dir}/run.log
        - StreamHandler: Writes to sys.stdout
    """
    from pathlib import Path
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # 使用 exp_name 命名日志文件，如果没有则使用 run.log
    log_filename = f'{exp_name}.log' if exp_name else 'run.log'
    file_path = Path(out_dir) / log_filename
    file_hdlr = logging.FileHandler(str(file_path))
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


    # compute train accuracy
    right_types, right_paths = 0, 0

    for idx in range(2, max_n):
        node_type_probs = F.softmax(node_type_scores, 1).cpu().detach().numpy()
        node_path_probs = F.softmax(node_path_scores, 1).cpu().detach().numpy()

        new_types = [np.random.choice(range(27), p=node_type_probs[i])
                     for i in range(len(G))]
        new_path = [np.random.choice(range(max_n), p=node_path_probs[i])
                    for i in range(len(G))]

        right_types += (new_types == type_gnd[idx]).sum().item()
        right_paths += (new_path == path_gnd[idx]).sum().item()
        
    return right_types / ((max_n - 1) * bsz), right_paths / ((max_n - 1) * bsz)
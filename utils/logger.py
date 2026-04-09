"""Logging utilities for training and evaluation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""

import sys
import logging

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
    return obj


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

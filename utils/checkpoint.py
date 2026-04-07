"""Checkpoint loading helpers.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
from utils.paths import resolve_path


def safe_load_checkpoint(checkpoint_path, map_location='cpu'):
    """Load a checkpoint with path resolution and a clear missing-file error."""
    resolved_path = resolve_path(checkpoint_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {resolved_path}\n"
            f"Original path provided: {checkpoint_path}\n"
            f"Make sure the checkpoint file exists at the specified location."
        )

    return torch.load(str(resolved_path), map_location=map_location)


def load_model_checkpoint(checkpoint_path, map_location='cpu'):
    """Load a model from either a model-only file or a legacy checkpoint dict."""
    checkpoint = safe_load_checkpoint(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        if 'model' not in checkpoint:
            raise KeyError(f"Legacy checkpoint is missing 'model': {checkpoint_path}")
        return checkpoint['model']

    return checkpoint

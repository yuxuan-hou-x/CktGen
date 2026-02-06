"""Evaluation configuration options.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import argparse
import os

def add_evaluate_options(parser):
    """
    Add evaluation and model checkpoint options.
    These options control model evaluation frequency, checkpoint loading, and inference settings.
    """
    group = parser.add_argument_group(description='Evaluation options')

    group.add_argument(
        '--infer_batch_size', 
        type=int, 
        default=128, 
        metavar='N', 
        help='Batch size for inference and evaluation tasks (reconstruction, prior validity, generation). '
             'Unlike training batch_size which is constrained by GPU memory during backpropagation, '
             'infer_batch_size can be larger (typically 128) for faster forward-only evaluation. '
             'Used in:' 
             '(1) reconstruction evaluation to batch test set circuits, '
             '(2) prior validity testing to batch random latent samples, '
             '(3) conditional generation to batch specification-to-circuit generation. '
             'Increase if you have more GPU memory; decrease if out-of-memory during evaluation. '
             'Default 128 works well for most GPUs with 8GB+ memory.'
    )
    group.add_argument(
        '--eval_interval', 
        type=int, 
        default=30, 
        metavar='N', 
        help='Evaluation frequency in epochs. The model will be evaluated on the test set every N epochs. '
             'Evaluation includes metrics like validity, spec accuracy, retrieval precision, FID, etc. '
             'Set to a smaller value for more frequent evaluation (slower training) or larger for faster training.'
    )
    group.add_argument(
        '--resume_pth', 
        type=str, 
        default=None, 
        help='Path to a checkpoint file (.pth) to resume training or run evaluation. '
             'When specified, the model weights will be loaded from this checkpoint. '
             'Example: "./output/cktgen/OCB101/filter/cktgen_cktarchi_checkpoint600.pth"'
    )
    group.add_argument(
        '--pretrained_eval_resume_pth', 
        type=str, 
        default=None, 
        help='Path to a pretrained evaluator model checkpoint for computing evaluation metrics. '
             'The evaluator is used to assess generated circuits by predicting their specifications '
             'and computing metrics like spec accuracy, retrieval precision, and FID. '
             'Required for conditional generation evaluation.'
    )
    group.add_argument(
        '--vae_pth', 
        type=str, 
        default=None, 
        help='Path to a pretrained VAE model checkpoint (.pth). Required for training LDT (Latent Diffusion Transformer) '
             'models, which use a frozen pretrained VAE to encode/decode circuits. The VAE parameters will not be updated '
             'during LDT training. Example: "./output/vae_cktarchi_checkpoint600.pth"'
    )
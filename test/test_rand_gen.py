"""Training script for VAE models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import json
import torch
import random
import numpy as np

import utils.logger as utils_logger
import utils.data as utils_data
import utils.paths as utils_paths

from torch import optim
from random import shuffle
from options.training import parser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.get_datasets import get_datasets
from models.get_model import get_model

import evaluation.reconstruct as eval_recon

def main():
    """Main entry point for training VAE models.
    
    Parses configuration, sets random seeds, initializes datasets and VAE model,
    configures optimizer and scheduler, then either resumes from checkpoint for
    evaluation or starts training from scratch.
    
    The function sets up:
        - Random seeds for reproducibility
        - Output directory for logs and checkpoints
        - Logger for experiment tracking
        - Dataset loaders (train/test splits)
        - VAE model with encoder architecture and decoder
        
    The trained VAE can be used:
        - Standalone for circuit reconstruction tasks
        - As a pretrained encoder for LDT (Latent Diffusion Transformer)
        - For learning compressed circuit representations
    """
    args = parser()

    seed = args['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ####--- Setup Paths (Centralized Path Management) ---####
    utils_paths.setup_paths(args)
    os.makedirs(args['out_dir'], exist_ok=True)

    ####--- Logger ---####
    logger = utils_logger.get_logger(args['out_dir'], args.get('exp_name'))
    logger.info(json.dumps(args, indent=4, sort_keys=True, default=utils_logger.serialize))

    ####--- Dataset & Network ---####
    datasets = get_datasets(args)
    model = get_model(args)

    logger.info('Training data length: %d'% (len(datasets['train'])))
    logger.info('Test data length: %d'% (len(datasets['test'])))
    logger.info('loading checkpoint from {}'.format(args['resume_pth']))
    
    ckpt = torch.load(args['resume_pth'], map_location='cpu')

    model = ckpt['model']   # 直接取出模型对象
    model = model.to(args['device'])
    # torch.save(model, 'output/cktgen_model_only.pth')

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Number of model parameters: %d'% (total_params))
    
    eval_recon.evaluate(args, model, datasets, logger)

if __name__ == "__main__":
    main()
"""Training script for CktGen conditional models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import json
import torch
import random
import time
import numpy as np

import utils.logger as utils_logger
import utils.data as utils_data
import utils.paths as utils_paths

from options.training import parser
from dataset.get_datasets import get_datasets
from models.get_model import get_model
import evaluation.auto_design as eval_auto_design
import evaluation.generate as eval_cond_gen

def main():
    """Main entry point for training CktGen models.
    
    Parses configuration, sets random seeds, initializes datasets and model,
    configures optimizer and scheduler, then either resumes from checkpoint
    for evaluation or starts training from scratch.
    
    The function sets up:
        - Random seeds for reproducibility
        - Output directory for logs and checkpoints
        - Logger for experiment tracking
        - Dataset loaders (train/test splits)
        - Model architecture based on parsed model name
        - AdamW optimizer with specified hyperparameters
        - Learning rate scheduler with plateau detection
        
    Training can be started fresh or resumed from a checkpoint if 'resume_pth' 
    is provided in args.
    """
    args = parser()

    ####--- Random seed ---####
    seed = args['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ####--- Setup Paths (Centralized Path Management) ---####
    # This ensures all paths are absolute and reliable regardless of where the script is run
    utils_paths.setup_paths(args)
    os.makedirs(args['out_dir'], exist_ok=True)

    ####--- Logger ---####
    logger = utils_logger.get_logger(args['out_dir'], args.get('exp_name'))
    logger.info(json.dumps(args, indent=4, sort_keys=True, default=utils_logger.serialize))
    
    ####--- Dataset & Network ---####
    datasets = get_datasets(args)
    
    # Load model from Hugging Face or local checkpoint
    if 'hf_model_id' in args and args['hf_model_id']:
        # Load from Hugging Face Hub
        logger.info('Loading model from Hugging Face Hub: {}'.format(args['hf_model_id']))
        wrapper = CktGenForConditionalGeneration.from_pretrained(args['hf_model_id'])
        model = wrapper.model
        # Update args with config from HF model
        config = wrapper.config
        for key, value in config.items():
            if key not in args:  # Don't override existing args
                args[key] = value
        logger.info('Model loaded from Hugging Face Hub successfully')
    else:
        # Load from local checkpoint (original behavior)
        model = get_model(args)
        logger.info('Training data length: %d, Test data length: %d'% (len(datasets['train']), len(datasets['test'])))
        logger.info('loading checkpoint from {}'.format(args['resume_pth']))
        model = torch.load(args['resume_pth'], map_location='cpu')
    
    model = model.to(args['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Number of model parameters: %d'% (total_params))
    
    eval_cond_gen.evaluate(args, model, datasets, logger)

if __name__ == "__main__":
    main()
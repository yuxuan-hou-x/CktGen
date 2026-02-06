"""Training script for Latent Diffusion Transformer.

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

from torch import optim
from random import shuffle
from options.training import parser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.get_datasets import get_datasets
from models.get_model import get_model

import evaluation.auto_design as eval_auto_design
import evaluation.generate as eval_cond_gen


def train(args, model, datasets, logger, optimizer, scheduler):
    """Main training loop for Latent Diffusion Transformer (LDT) models.
    
    Trains a diffusion model in the latent space of a pre-trained VAE. The VAE 
    encoder maps circuits to latents, then LDT learns to denoise/generate latents
    conditioned on specifications. This two-stage approach is more efficient than
    diffusing in the original circuit space.
    
    Args:
        args: Configuration dictionary with training hyperparameters including:
            - 'epochs': Maximum training epochs
            - 'batch_size': Mini-batch size
            - 'save_interval': Checkpoint saving frequency (epochs)
            - 'eval_interval': Evaluation frequency (epochs)
        model: LDT model with pre-trained VAE and transformer denoiser.
        datasets: Dictionary with 'train' and 'test' dataset splits.
        logger: Logger instance for recording training progress.
        optimizer: AdamW optimizer for LDT parameters (VAE frozen).
        scheduler: ReduceLROnPlateau learning rate scheduler.
        
    Notes:
        - VAE is frozen during training; only LDT parameters are optimized
        - Encodes circuits to latent space using VAE encoder before training
        - Samples from latent distribution (not mean) for better coverage
        - Evaluates on conditional generation and auto-design tasks periodically
        - Requires pre-trained VAE checkpoint specified in args['vae_pth']
    """
    logger.info('###############################################################################')
    logger.info('                                Training Start')
    logger.info('###############################################################################')
    
    total_training_time = 0.0
    
    for epoch in range(1, args['epochs']+1):
        shuffle(datasets['train'])
        print_iter = 1
        batch_graphs = []
        train_loss = 0.0
        time_start = time.time()
        
        avg_train_loss = 0.0
        
        for i, g in enumerate(datasets['train']):
            model.train()
            batch_graphs.append(g)
            if len(batch_graphs) == args['batch_size'] or i == len(datasets['train']) - 1:
                optimizer.zero_grad()
                bsz = len(batch_graphs)
                
                _batch_graphs = utils_data.collate_fn(batch_graphs)
                batch = utils_data.transforms(args, _batch_graphs)

                batch.update(model.vae.archi.encode(batch))
                batch['ckt_latents'] = model.vae.sample_from_distribution(batch['ckt_dists'], sample_mean=False)

                loss            = model.compute_loss(batch)
                avg_train_loss  += loss

                train_loss += float(loss.item())

                # if print_iter % args['print_iter'] == 0:
                #     logger.info('Train. Iter: %d , loss: %0.4f'% (
                #                 print_iter,
                #                 avg_train_loss / (bsz * print_iter), 
                #             )
                #         )
                loss.backward()
                optimizer.step()
                batch_graphs = []
                print_iter = print_iter + 1
            else:
                continue

        time_end = time.time()
        epoch_time = time_end - time_start
        total_training_time += epoch_time

        logger.info('Epoch: %d, loss: %0.4f, Time: %.2f s'% (
                epoch, 
                avg_train_loss / len(datasets['train']),
                epoch_time
            )
        )

        scheduler.step(train_loss)

        if (epoch % args['save_interval'] == 0) or (epoch == args['save_interval']):
            checkpoint_path = utils_paths.get_checkpoint_path(args['out_dir'], args['exp_name'], epoch)
            torch.save(model, checkpoint_path)

        if (epoch % args['eval_interval'] == 0) or (epoch == args['epochs']):
            eval_cond_gen.evaluate(args, model, datasets, logger)
            eval_auto_design.evaluate(args, model, datasets, logger)
    
    logger.info('###############################################################################')
    logger.info('                                Training Complete')
    logger.info('###############################################################################')
    logger.info('Average Time per Epoch: %.2f s'% (total_training_time / args['epochs']))


def main():
    """Main entry point for training Latent Diffusion Transformer models.
    
    Parses configuration, sets random seeds, initializes datasets and LDT model,
    loads pre-trained VAE, configures optimizer for LDT parameters only, then
    either resumes from checkpoint for evaluation or starts training from scratch.
    
    The function sets up:
        - Random seeds for reproducibility
        - Output directory for logs and checkpoints
        - Logger for experiment tracking
        - Dataset loaders (train/test splits)
        - LDT model with frozen pre-trained VAE
        - AdamW optimizer ONLY for LDT parameters (excludes VAE)
        - Learning rate scheduler with plateau detection
        
    Key requirements:
        - Pre-trained VAE checkpoint must be provided via args['vae_pth']
        - VAE is loaded and frozen; only diffusion transformer is trained
        - Optimizer filters parameters to exclude 'vae.*' from updates
        
    Raises:
        ValueError: If 'vae_pth' is not provided in args.
    """
    args = parser()

    ####--- Random seed ---####
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
    
    logger.info('Training data length: %d, Test data length: %d'% (len(datasets['train']), len(datasets['test'])))

    ####--- Optimizer & Scheduler ---####
    ldt_params = [v for k, v in model.named_parameters() if not k.startswith('vae.')]
    optimizer = optim.AdamW(
        ldt_params, 
        lr=args['lr'], 
        betas=(args['beta'][0], args['beta'][1]), 
        weight_decay=args['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    ####--- Load VAE ---####
    if 'vae_pth' not in args:
        raise ValueError('vae_pth is required in args for training LDT')
    model.load_vae(args['vae_pth'])

    train(args, model, datasets, logger, optimizer, scheduler)

if __name__ == "__main__":
    main()
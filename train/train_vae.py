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

def train(args, model, datasets, logger, optimizer, scheduler):
    """Main training loop for VAE (Variational Autoencoder) models.
    
    Trains an unconditional VAE to encode circuit graphs into a latent space and
    reconstruct them. The model learns a compressed latent representation while
    balancing reconstruction accuracy and latent space regularization (KL divergence).
    
    Args:
        args: Configuration dictionary with training hyperparameters including:
            - 'epochs': Maximum training epochs
            - 'batch_size': Mini-batch size
            - 'save_interval': Checkpoint saving frequency (epochs)
            - 'eval_interval': Evaluation frequency (epochs)
        model: VAE model with encoder (architecture) and decoder.
        datasets: Dictionary with 'train' and 'test' dataset splits.
        logger: Logger instance for recording training progress.
        optimizer: AdamW optimizer for all model parameters.
        scheduler: ReduceLROnPlateau learning rate scheduler.
        
    Notes:
        - Samples from latent distribution (not mean) during training for better coverage
        - Total loss = reconstruction loss + KL divergence loss
        - Reconstruction loss includes: type, path, size, and edge prediction losses
        - KL loss regularizes the latent space to be close to a standard normal
        - Saves full checkpoint dict with model, optimizer, and scheduler states
        - Evaluates reconstruction quality periodically on test set
    """
    logger.info('###############################################################################')
    logger.info('                                Training Start')
    logger.info('###############################################################################')
    for epoch in range(1, args['epochs']+1):
        shuffle(datasets['train'])
        batch_graphs = []
        train_loss = 0.0
        (avg_train_loss, avg_recon_loss, avg_kl_divergence_loss, avg_type_loss, avg_path_loss, 
        avg_size_loss, avg_edge_loss) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


        for i, g in enumerate(datasets['train']):
            model.train()
            batch_graphs.append(g)
            if len(batch_graphs) == args['batch_size'] or i == len(datasets['train']) - 1:
                optimizer.zero_grad()
                bsz = len(batch_graphs)
                
                _batch_graphs = utils_data.collate_fn(batch_graphs)
                batch = utils_data.transforms(args, _batch_graphs)

                batch.update(model.archi.encode(batch))
                batch['ckt_latents'] = model.sample_from_distribution(batch['ckt_dists'], sample_mean=False)

                mixed_loss, losses      = model.compute_loss(batch)

                avg_train_loss          += mixed_loss.item()
                avg_recon_loss          += losses['recon'].item()
                avg_kl_divergence_loss  += losses['kl'].item()
                avg_type_loss           += losses['types'].item()
                avg_path_loss           += losses['paths'].item()
                avg_size_loss           += losses['sizes'].item()
                avg_edge_loss           += losses['edges'].item()
                train_loss              += float(mixed_loss.item())
                
                mixed_loss.backward()
                optimizer.step()
                batch_graphs = []

            else:
                continue

        logger.info('Epoch: %d, Total: %0.4f, rec: %0.4f, kl_d: %0.4f, types: %0.4f, paths: %0.4f, sizes: %0.4f, edges: %0.4f'% (
                    epoch, 
                    avg_train_loss / len(datasets['train']), 
                    avg_recon_loss / len(datasets['train']), 
                    avg_kl_divergence_loss / len(datasets['train']), 
                    avg_type_loss / len(datasets['train']),
                    avg_path_loss / len(datasets['train']),
                    avg_size_loss / len(datasets['train']),
                    avg_edge_loss / len(datasets['train'])))

        scheduler.step(train_loss)


        if (epoch % args['save_interval'] == 0) or (epoch == args['save_interval']):
            checkpoint_path = utils_paths.get_checkpoint_path(args['out_dir'], args['exp_name'], epoch)
            torch.save(model, checkpoint_path)
            # torch.save({'model': model,
            #             'model_state': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()},
            #             checkpoint_path)


        if (epoch % args['eval_interval'] == 0) or (epoch == args['epochs']):
            eval_recon(args, model, datasets, logger)


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
        - AdamW optimizer with specified hyperparameters
        - Learning rate scheduler with plateau detection
        
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

    ####--- Optimizer & Scheduler ---####
    optimizer = optim.AdamW(model.parameters(), 
                            lr=args['lr'], 
                            betas=(args['beta'][0], args['beta'][1]), 
                            weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train(args, model, datasets, logger, optimizer, scheduler)

if __name__ == "__main__":
    main()
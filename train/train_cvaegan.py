"""Training script for Conditional VAE-GAN models.

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
    """Main training loop for Conditional VAE-GAN (CVAEGAN) models.
    
    Alternates between training the CVAE (encoder-decoder) and discriminator networks.
    The CVAE learns to reconstruct circuits conditioned on specifications, while the
    discriminator distinguishes real circuits from generated ones.
    
    Args:
        args: Configuration dictionary with training hyperparameters including:
            - 'epochs': Maximum training epochs
            - 'batch_size': Mini-batch size  
            - 'save_interval': Checkpoint saving frequency (epochs)
            - 'eval_interval': Evaluation frequency (epochs)
            - 'device': Torch device for computation
        model: CVAEGAN model with cvae and discriminator components.
        datasets: Dictionary with 'train' and 'test' dataset splits.
        logger: Logger instance for recording training progress.
        optimizer: Dictionary with 'cvae' and 'discriminator' optimizers.
        scheduler: Dictionary with 'cvae' and 'discriminator' schedulers.
        
    Notes:
        - Training alternates: first CVAE step, then discriminator step
        - CVAE losses: reconstruction (types, paths, sizes, edges) + KL divergence
        - Discriminator losses: real circuit classification + fake circuit classification
        - Real labels = 1, fake labels = 0 for adversarial training
        - Saves full model checkpoint at specified intervals
        - Evaluates on conditional generation and auto-design tasks periodically
    """
    logger.info('###############################################################################')
    logger.info('                                Training Start')
    logger.info('###############################################################################')
    
    total_training_time = 0.0
    
    for epoch in range(1, args['epochs']+1):
        shuffle(datasets['train'])
        batch_graphs = []
        time_start = time.time()

        train_discriminator_loss = 0.0
        train_cvae_loss = 0.0

        (avg_recon_loss, avg_kl_loss, avg_type_loss, avg_path_loss, avg_size_loss, avg_edge_loss, 
        avg_adver_loss, avg_real_loss, avg_fake_loss) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        for i, g in enumerate(datasets['train']):
            model.train()
            batch_graphs.append(g)

            if len(batch_graphs) == args['batch_size'] or i == len(datasets['train']) - 1:
                bsz = len(batch_graphs)
                
                _batch_graphs = utils_data.collate_fn(batch_graphs)
                batch = utils_data.transforms(args, _batch_graphs)

                batch = {
                    key: val.to(args['device']) 
                    if torch.is_tensor(val) else val 
                    for key, val in batch.items()
                }

                ##############################################################################################################
                #  Train VAE
                ##############################################################################################################
                optimizer['cvae'].zero_grad()

                spec_embs = model.spec_embedder(batch['gains'], batch['bws'], batch['pms'])
                batch['spec_embs'] = model.spec_project(spec_embs)

                batch['ckt_latents'], batch['ckt_dists'] = model.sample_ckt_latents(batch, sample_mean=False, return_dists=True)
                
                mixed_loss, losses = model.compute_cvae_loss(batch)

                avg_recon_loss  += losses['recon'].item()
                avg_type_loss   += losses['types'].item()
                avg_path_loss   += losses['edges'].item()
                avg_size_loss   += losses['sizes'].item()
                avg_edge_loss   += losses['edges'].item()
                avg_kl_loss     += losses['kl'].item()

                train_cvae_loss += float(mixed_loss.item())

                mixed_loss.backward()
                optimizer['cvae'].step()

                ##############################################################################################################
                #  Train Discriminator
                ##############################################################################################################
    
                optimizer['discriminator'].zero_grad()
                
                real_labels = torch.ones(bsz, 1).to(args['device'])
                fake_labels = torch.zeros(bsz, 1).to(args['device'])
                
                ####--- ground-truth discriminate ---####
                loss_discriminate_gnd = model.compute_discriminator_loss(batch, real_labels)

                ####--- reconstruct discriminate ---####
                gen_ckts = model(args, batch)
                batch_gen = utils_data.transform_cktarchi(args, gen_ckts)
                batch_gen.update(utils_data.transform_specification(args, _batch_graphs))
                batch_gen = {
                    key: val.to(args['device']) 
                    if torch.is_tensor(val) else val 
                    for key, val in batch.items()
                }
                loss_discriminate_gen = model.compute_discriminator_loss(batch_gen, fake_labels)
                
                adver_loss = (loss_discriminate_gen + loss_discriminate_gnd) / 2
                avg_adver_loss += adver_loss
                
                adver_loss.backward()
                optimizer['discriminator'].step()

                train_discriminator_loss += float(adver_loss.item())

                batch_graphs = []
            else:
                continue

        time_end = time.time()
        epoch_time = time_end - time_start
        total_training_time += epoch_time

        # avg_train_loss = (avg_recon_loss + avg_adver_loss + avg_real_loss + avg_fake_loss) 
        logger.info('Epoch: %d, kl: %0.4f, recon: %0.4f, types: %0.4f, paths: %0.4f, sizes: %0.4f, edges: %0.4f, adver: %0.4f, Time: %.2f s'% (
            epoch, 
            avg_kl_loss / len(datasets['train']), 
            avg_recon_loss / len(datasets['train']), 
            avg_type_loss / len(datasets['train']),
            avg_path_loss / len(datasets['train']),
            avg_size_loss / len(datasets['train']),
            avg_edge_loss / len(datasets['train']),
            avg_adver_loss / len(datasets['train']),
            epoch_time
        ))

        scheduler['discriminator'].step(train_discriminator_loss)
        scheduler['cvae'].step(train_cvae_loss)

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
    """Main entry point for training CVAEGAN models.
    
    Parses configuration, sets random seeds, initializes datasets and model,
    configures separate optimizers for CVAE and discriminator, then either 
    resumes from checkpoint for evaluation or starts training from scratch.
    
    The function sets up:
        - Random seeds for reproducibility  
        - Output directory for logs and checkpoints
        - Logger for experiment tracking
        - Dataset loaders (train/test splits)
        - CVAEGAN model with encoder, decoder, and discriminator
        - Separate AdamW optimizers for CVAE and discriminator
        - Separate learning rate schedulers for both components
        
    Key differences from standard training:
        - Dual optimizer setup (one for CVAE, one for discriminator)
        - Dual scheduler setup with independent plateau detection
        - Adversarial training requires careful balance between components
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
    optimizer, scheduler = {}, {}

    cvae_params = [v for k, v in model.named_parameters() if not k.startswith('discriminator.')]
    optimizer['cvae'] = optim.AdamW(cvae_params, lr=args['lr'], betas=(args['beta'][0], args['beta'][1]), weight_decay=args['weight_decay'])
    scheduler['cvae'] = ReduceLROnPlateau(optimizer['cvae'], 'min', factor=0.1, patience=10, verbose=True)

    optimizer['discriminator'] = optim.AdamW(model.discriminator.parameters(), lr=args['lr'], betas=(args['beta'][0], args['beta'][1]), weight_decay=args['weight_decay'])
    scheduler['discriminator'] = ReduceLROnPlateau(optimizer['discriminator'], 'min', factor=0.1, patience=10, verbose=True)

    train(args, model, datasets, logger, optimizer, scheduler)


if __name__ == "__main__":
    main()
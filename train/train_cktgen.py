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

from torch import optim
from random import shuffle
from options.training import parser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.get_datasets import get_datasets
from models.get_model import get_model

import evaluation.auto_design as eval_auto_design
import evaluation.generate as eval_cond_gen

def train(args, model, datasets, logger, optimizer, scheduler):
    """Main training loop for CktGen conditional generative models.
    
    Trains the model using batched mini-batch gradient descent with various loss components
    including reconstruction, KL divergence, alignment, contrastive (NCE), and guided losses.
    Periodically saves checkpoints and evaluates on conditional generation and auto-design tasks.
    
    Args:
        args: Configuration dictionary containing training hyperparameters including:
            - 'epochs': Maximum training epochs
            - 'batch_size': Mini-batch size
            - 'save_interval': Checkpoint saving frequency (epochs)
            - 'eval_interval': Evaluation frequency (epochs)
            - 'vae': Whether using VAE (requires KL loss)
            - 'conditioned': Whether model is conditioned on specifications
            - 'contrastive': Whether to use NCE contrastive loss
            - 'guided': Whether to use guided diffusion loss
        model: CktGen model instance to train.
        datasets: Dictionary with 'train' and 'test' dataset splits.
        logger: Logger instance for recording training progress.
        optimizer: AdamW optimizer for model parameters.
        scheduler: ReduceLROnPlateau learning rate scheduler.
        
    Notes:
        - Shuffles training data at the start of each epoch
        - Accumulates gradients per batch then updates
        - Logs per-epoch losses for all active loss components
        - Saves model checkpoints at specified intervals
        - Runs evaluation on conditional generation and auto-design tasks periodically
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
        
        (avg_train_loss, avg_recon_loss, avg_kl_loss, avg_align_loss, avg_nce_loss, avg_gde_loss, 
        avg_type_loss, avg_path_loss, avg_size_loss, avg_edge_loss) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        for i, g in enumerate(datasets['train']):
            model.train()
            batch_graphs.append(g)
            if len(batch_graphs) == args['batch_size'] or i == len(datasets['train']) - 1:
                optimizer.zero_grad()
                bsz = len(batch_graphs)
                
                _batch_graphs = utils_data.collate_fn(batch_graphs)
                batch = utils_data.transforms(args, _batch_graphs)
                
                if args['vae']:
                    batch['ckt_latents'], batch['ckt_dists'] = model.sample_ckt_latents(batch, sample_mean=False, return_dists=True)
                else:
                    batch['ckt_latents'] = model.sample_ckt_latents(batch, sample_mean=False, return_dists=False)

                mixed_loss, losses      = model.compute_loss(batch)
                avg_train_loss          += mixed_loss.item()
                avg_recon_loss          += losses['recon'].item()
                avg_type_loss           += losses['types'].item()
                avg_path_loss           += losses['paths'].item()
                avg_size_loss           += losses['sizes'].item()
                avg_edge_loss           += losses['edges'].item()

                if args['conditioned']:
                    avg_align_loss += losses['align'].item()
                    if args['contrastive']:
                        avg_nce_loss += losses['nce'].item()
                    if args['guided']:
                        avg_gde_loss += losses['gde'].item()
                    
                if args['vae']:
                    avg_kl_loss += losses['kl'].item()

                train_loss += float(mixed_loss.item())

                # acc_type                = losses['acc_type']
                # acc_path                = losses['acc_path']
                # acc_edge                = losses['acc_edge']

                # if print_iter % args['print_iter'] == 0:
                #     logger.info('Train. Iter %d :, Total: %0.4f, rec: %0.4f, kl: %0.4f, types: %0.4f, paths: %0.4f, sizes: %0.4f, edges: %0.4f, acc_type: %0.4f, acc_path: %0.4f, acc_edge: %0.4f'% (
                #                 print_iter,
                #                 avg_train_loss / (bsz * print_iter), 
                #                 avg_recon_loss / (bsz * print_iter), 
                #                 avg_kl_loss / (bsz * print_iter), 
                #                 avg_type_loss / (bsz * print_iter), 
                #                 avg_path_loss / (bsz * print_iter),
                #                 avg_size_loss / (bsz * print_iter),
                #                 avg_edge_loss / (bsz * print_iter),
                #                 acc_type, acc_path, acc_edge))
                
                mixed_loss.backward()
                optimizer.step()
                batch_graphs = []
                print_iter = print_iter + 1

            else:
                continue

        time_end = time.time()
        epoch_time = time_end - time_start
        total_training_time += epoch_time

        logger.info('Epoch: %d, Total: %0.4f, kl: %0.4f, align: %0.4f, nce: %0.4f, gde: %0.4f, recon: %0.4f, types: %0.4f, paths: %0.4f, sizes: %0.4f, edges: %0.4f, Time: %.2f s'% (
            epoch, 
            avg_train_loss / len(datasets['train']), 
            avg_kl_loss / len(datasets['train']), 
            avg_align_loss / len(datasets['train']), 
            avg_nce_loss / len(datasets['train']), 
            avg_gde_loss / len(datasets['train']), 
            avg_recon_loss / len(datasets['train']), 
            avg_type_loss / len(datasets['train']),
            avg_path_loss / len(datasets['train']),
            avg_size_loss / len(datasets['train']),
            avg_edge_loss / len(datasets['train']),
            epoch_time
        ))
        
        scheduler.step(train_loss)

        if (epoch % args['save_interval'] == 0) or (epoch == args['save_interval']):
            checkpoint_path = utils_paths.get_checkpoint_path(args['out_dir'], args['exp_name'], epoch)
            torch.save(model, checkpoint_path)
            # torch.save({'model': model,
            #             'model_state': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()},
            #             os.path.join(args['out_dir'], '{}_checkpoint{}.pth'.format(args['modelname'], epoch)))

        if (epoch % args['eval_interval'] == 0) or (epoch == args['epochs']):
            eval_cond_gen.evaluate(args, model, datasets, logger)
            eval_auto_design.evaluate(args, model, datasets, logger)
            # if args['conditioned']:
            #     eval_gen(args, datasets, logger, model)
            # else:
            #     eval_recon(args, model, datasets, logger)
    
    logger.info('###############################################################################')
    logger.info('                                Training Complete')
    logger.info('###############################################################################')
    logger.info('Average Time per Epoch: %.2f s'% (total_training_time / args['epochs']))

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
    model = get_model(args)

    logger.info('Training data length: %d, Test data length: %d'% (len(datasets['train']), len(datasets['test'])))

    ####--- Optimizer & Scheduler ---####
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
        betas=(args['beta'][0], args['beta'][1]),
        weight_decay=args['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)


    train(args, model, datasets, logger, optimizer, scheduler)

if __name__ == "__main__":
    main()
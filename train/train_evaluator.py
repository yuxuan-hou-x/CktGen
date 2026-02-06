"""Training script for performance evaluator models.

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
from scipy.stats import pearsonr
from models.get_model import get_model

import evaluation.retrieval as eval_retri

def RMSE(pred, gnd):
    """Computes Root Mean Square Error between predictions and ground truth.
    
    Args:
        pred: Predicted values (numpy array or tensor).
        gnd: Ground truth values (numpy array or tensor).
        
    Returns:
        float: RMSE value.
    """
    # return torch.sqrt(torch.mean((pred - gnd) ** 2))
    return np.sqrt(np.mean((pred - gnd) ** 2))

@torch.no_grad()
def eval_pred(args, model, datasets, logger):
    """Evaluates performance predictor on test set.
    
    Computes circuit embeddings, predicts performance metrics (gain, bandwidth, 
    phase margin, FoM), and calculates RMSE and Pearson correlation for each metric.
    
    Args:
        args: Configuration dictionary with device and normalization parameters.
        model: Evaluator model with circuit encoder and performance predictor.
        datasets: Dictionary containing 'test' dataset.
        logger: Logger for recording evaluation metrics.
        
    Notes:
        - Uses model's training mean/std for denormalization
        - Evaluates on gain, bandwidth (bw), phase margin (pm), and figure-of-merit (fom)
        - Reports both RMSE and Pearson correlation (P-R) for each metric
    """
    model.eval()

    args['means'] = model.mean_train
    args['stds'] = model.std_train


    batch = utils_data.transforms(args, datasets['test'])
    ckt_embs = model.get_ckt_embeddings(batch)
    gnds = utils_data.transform_continues_specification(args, datasets['test'])

    preds = model.predict_standard(ckt_embs)

    pred_gain = preds['gain'].cpu().numpy()
    pred_bw = preds['bw'].cpu().numpy()
    pred_pm = preds['pm'].cpu().numpy()
    pred_fom = preds['fom'].cpu().numpy()

    gnd_gain = gnds['cont_gains'].cpu().numpy()
    gnd_bw = gnds['cont_gains'].cpu().numpy()
    gnd_pm = gnds['cont_gains'].cpu().numpy()
    gnd_fom = gnds['cont_gains'].cpu().numpy()

    rmse_gain  = RMSE(pred_gain, gnd_gain)
    rmse_bw    = RMSE(pred_bw, gnd_bw)
    rmse_pm    = RMSE(pred_pm, gnd_pm)
    rmse_fom   = RMSE(pred_fom, gnd_fom)

    r_gain, _   = pearsonr(pred_gain, gnd_gain)
    r_bw, _     = pearsonr(pred_bw, gnd_bw)
    r_pm, _     = pearsonr(pred_pm, gnd_pm)
    r_fom, _    = pearsonr(pred_fom, gnd_fom)
    

    logger.info('RMSE Gain: %0.4f, P-R Gain: %0.4f, RMSE BW: %0.4f, P-R BW: %0.4f, RMSE PM: %0.4f, P-R PM: %0.4f, RMSE FoM: %0.4f, P-R FoM: %0.4f'% (
        rmse_gain,
        rmse_bw,
        rmse_pm,
        rmse_fom,
        r_gain,
        r_bw,
        r_pm,
        r_fom,
    ))


def train(args, model, datasets, logger, optimizer, scheduler):
    """Main training loop for performance evaluator models.
    
    Trains a model to predict circuit performance metrics from circuit embeddings
    using a combination of NCE contrastive loss, alignment loss, and prediction loss.
    The evaluator learns joint embeddings of circuits and specifications for retrieval.
    
    Args:
        args: Configuration dictionary with training hyperparameters including:
            - 'epochs': Maximum training epochs
            - 'batch_size': Mini-batch size
            - 'save_interval': Checkpoint saving frequency (epochs)
            - 'eval_interval': Evaluation frequency (epochs)
        model: Evaluator model with circuit encoder, spec encoder, and predictor.
        datasets: Dictionary with 'train' and 'test' dataset splits.
        logger: Logger instance for recording training progress.
        optimizer: AdamW optimizer for model parameters.
        scheduler: ReduceLROnPlateau learning rate scheduler.
        
    Notes:
        - Normalizes FoM values using training set mean and standard deviation
        - Computes three loss components: NCE (contrastive), align, and pred (prediction)
        - Saves checkpoints and evaluates on retrieval task periodically
        - FoM standardization helps stabilize training
    """
    logger.info('###############################################################################')
    logger.info('                                Training Start')
    logger.info('###############################################################################')

    mean, std = utils_data.get_fom_train_mean_and_std(args, datasets['train'])
    model.set_train_mean_std(mean, std)

    for epoch in range(1, args['epochs']+1):
        shuffle(datasets['train'])
        print_iter = 1
        batch_graphs = []
        train_loss = 0.0
        time_start = time.time()
        
        avg_train_loss, avg_align_loss, avg_nce_loss, avg_pred_loss  = 0.0, 0.0, 0.0, 0.0
        
        for i, g in enumerate(datasets['train']):
            model.train()
            batch_graphs.append(g)

            if len(batch_graphs) == args['batch_size'] or i == len(datasets['train']) - 1:
                optimizer.zero_grad()
                bsz = len(batch_graphs)
                
                _batch_graphs = utils_data.collate_fn(batch_graphs)
                batch = utils_data.transforms(args, _batch_graphs)
                batch['foms'] = utils_data.standard_fom(model.fom_train_mean, model.fom_train_std, batch['foms'])
                
                batch['ckt_embs'] = model.get_ckt_embeddings(batch)
                batch['spec_embs'] = model.get_spec_embeddings(batch) # .squeeze(0)

                mixed_loss, losses      = model.compute_loss(batch)
                avg_train_loss          += mixed_loss.item()
                avg_nce_loss            += losses['nce'].item()
                avg_align_loss          += losses['align'].item()
                avg_pred_loss           += losses['pred'].item()
                        
                train_loss += float(mixed_loss.item())
                mixed_loss.backward()

                optimizer.step()
                batch_graphs = []
                print_iter = print_iter + 1
            else:
                continue


        logger.info('Epoch: %d, Total: %0.4f, nce: %0.4f, align: %0.4f, pred: %0.4f'% (
                    epoch, 
                    avg_train_loss / len(datasets['train']), 
                    avg_nce_loss / len(datasets['train']), 
                    avg_align_loss / len(datasets['train']),
                    avg_pred_loss / len(datasets['train'])
                ))

        scheduler.step(mixed_loss)

        if (epoch % args['save_interval'] == 0) or (epoch == args['save_interval']):
            checkpoint_path = utils_paths.get_checkpoint_path(args['out_dir'], args['exp_name'], epoch)
            torch.save(model, checkpoint_path)

        if (epoch % args['eval_interval'] == 0) or (epoch == args['epochs']):
            eval_retri.evaluate(args, model, datasets, logger)

def main():
    """Main entry point for training evaluator models.
    
    Parses configuration, sets random seeds, initializes datasets and evaluator model,
    configures optimizer and scheduler, then either resumes from checkpoint for 
    evaluation or starts training from scratch.
    
    The function sets up:
        - Random seeds for reproducibility
        - Output directory for logs and checkpoints
        - Logger for experiment tracking
        - Dataset loaders (train/test splits)
        - Evaluator model for performance prediction and retrieval
        - AdamW optimizer with specified hyperparameters
        - Learning rate scheduler with plateau detection
        
    The evaluator is used for:
        - Predicting circuit performance metrics (gain, bw, pm, fom)
        - Learning joint embeddings for circuit-specification retrieval
        - Supporting automated design through performance-guided search
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
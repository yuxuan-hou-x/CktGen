"""Circuit reconstruction evaluation metrics.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import time
import torch
import utils.data as utils_data
from tqdm import tqdm
from evaluation.tools import is_same_DAG, is_valid_DAG, is_valid_Circuit, ratio_same_DAG
import evaluation.prior_validity as eval_prior_validity

@torch.no_grad()
def evaluate(args, model, datasets, logger):
    """Evaluates circuit reconstruction and unconditional generation quality.
    
    Tests the model's ability to:
    1. Reconstruct circuits from their encoded latent representations
    2. Generate valid/novel circuits from random prior sampling
    
    Reconstruction evaluation:
    - Encodes test circuits to latent space
    - Decodes back to circuit graphs
    - Computes reconstruction loss and perfect reconstruction accuracy
    
    Prior sampling evaluation:
    - Calls eval_prior_validity.evaluate() for unconditional generation
    - Measures validity (DAG/circuit) and novelty
    
    Args:
        args: Configuration with 'infer_batch_size', 'device', 'archiname'.
        model: VAE/generative model with encode/decode methods.
        datasets: Dictionary with 'test' and 'train' datasets.
        logger: Logger instance.
        
    Side Effects:
        - Logs comprehensive reconstruction and generation metrics
        - Prints evaluation summary with loss, accuracy, validity, novelty
        
    Notes:
        - Uses sample_mean=True for deterministic reconstruction
        - Combines reconstruction metrics with prior validity metrics
    """
    
    # test recon accuracy
    model.eval()

    # sample_times, decode_times = 10, 10
    sample_times, decode_times = 1, 1
    avg_rec_loss, pred_loss = 0, 0
    rec_acc = 0
    num_perfect_rec = 0
    
    # print_iter = 1
    gnd_graphs = []
    
    tot_time = 0
    time_start = time.time()
    logger.info('###############################################################################')
    logger.info('                  Starting Reconstruction Evaluation')
    logger.info('###############################################################################')
    logger.info('Reconstructing %d test circuits...' % len(datasets['test']))
    
    for i, g in enumerate(tqdm(datasets['test'], desc="Reconstructing circuits")):
        gnd_graphs.append(g)

        if len(gnd_graphs) == args["infer_batch_size"] or i == len(datasets['test']) - 1:
            _gnd_graphs = utils_data.collate_fn(gnd_graphs)
            batch = utils_data.transforms(args, _gnd_graphs)
            batch.update(model.archi.encode(batch)) # encode and get mu, logvar token
            batch['ckt_latents'] = model.sample_from_distribution(batch['ckt_dists'], sample_mean=True)

            mixed_loss, recon_losses = model.compute_loss(batch)
            avg_rec_loss += mixed_loss.item()
            for _ in range(sample_times):
                batch['ckt_latents'] = model.sample_from_distribution(batch['ckt_dists'], sample_mean=True)
                for _ in range(decode_times):
                    rec_graphs = model.archi.decode(args, batch)    
                    num_perfect_rec += sum(is_same_DAG(g_gnd, g_rec)
                                           for g_gnd, g_rec
                                           in zip(gnd_graphs, rec_graphs))  # compare reconstruct to gnd-truth

            # logger.info('Test. Iter: %d, Reconstruct Loss : %.04f'% (print_iter, (mixed_loss.item() / len(gnd_graphs))))

            gnd_graphs = []
            # print_iter = print_iter + 1
    
    avg_rec_loss /= len(datasets['test'])
    # time_end = time.time()
    # tot_time += time_end - time_start
    # comp_time = time_end - time_start
    rec_acc = num_perfect_rec / (len(datasets['test']) * sample_times * decode_times)
    
    logger.info('###############################################################################')
    logger.info('                  Starting Unconditional Generation Evaluation')
    logger.info('###############################################################################')
    rand_gen_res = eval_prior_validity.evaluate(args, model, datasets, logger, scale_to_train_range=True)

    logger.info('###############################################################################')
    logger.info('                  Reconstruct and Unconditional Generation Evaluation')
    logger.info('###############################################################################')
    logger.info(
        'Average Reconstruct Loss : %.04f, Reconstruct Accuracy: %0.04f, Infer time: %.04f, Valid DAG: %0.04f, Valid Circuit: %0.04f, Novel Circuits: %0.04f'% (
            avg_rec_loss, 
            rec_acc,
            rand_gen_res['infer_time'],
            rand_gen_res['rto_valid_dag'],
            rand_gen_res['rto_valid_ckt'],
            rand_gen_res['rto_novel_dag']
        )
    )
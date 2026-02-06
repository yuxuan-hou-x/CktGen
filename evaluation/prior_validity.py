"""Validity evaluation for circuits sampled from prior.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import time
import torch
import utils.data as utils_data
from tqdm import tqdm
from evaluation.tools import is_same_DAG, is_valid_DAG, is_valid_Circuit, ratio_same_DAG, extract_latents

@torch.no_grad()
def evaluate(args, model, datasets, logger, scale_to_train_range=False):
    """Evaluates validity and novelty of circuits generated from random prior sampling.
    
    Samples random latent vectors from Gaussian prior, decodes them to circuits,
    and evaluates:
    - Valid DAG ratio: Percentage of valid directed acyclic graphs
    - Valid circuit ratio: Percentage meeting circuit topology constraints
    - Novel circuit ratio: Percentage not appearing in training set
    
    Args:
        args: Configuration with 'device', 'latent_dim', 'infer_batch_size', 'archiname'.
        model: VAE/generative model with archi.decode method.
        datasets: Dictionary with 'train' dataset for novelty comparison.
        logger: Logger instance.
        scale_to_train_range: If True, scales random latents to match training
                             distribution (mean, std). Default False.
                             
    Returns:
        dict: Evaluation metrics with keys:
            - 'infer_time': Total inference time (seconds)
            - 'rto_valid_ckt': Ratio of valid circuits
            - 'rto_valid_dag': Ratio of valid DAGs
            - 'rto_novel_dag': Ratio of novel DAGs (not in training set)
            
    Notes:
        - Samples 1000 latent points, decodes each 10 times
        - Total evaluations: 10,000 circuits
    """
    model.eval()
    
    if scale_to_train_range:
        train_range = extract_latents(args, model, datasets['train'])
        mean, std = train_range.mean(0), train_range.std(0)
        mean, std = torch.FloatTensor(mean).to(args['device']), torch.FloatTensor(std).to(args['device'])

    n_latent_points = 1000
    decode_times = 10

    num_valid_dags, num_valid_ckts = 0, 0
    valid_dags, valid_ckts = [], []
    time_start = time.time()
    cnt = 0
    logger.info('Generating %d circuits from %d latent points (decode_times=%d)...' % 
                (n_latent_points * decode_times, n_latent_points, decode_times))
    
    for i in tqdm(range(n_latent_points), desc="Unconditional generation"):
        cnt += 1
        # logger.info('latent points: %d '% (i))
        if cnt == args["infer_batch_size"] or i == n_latent_points - 1:
            batch = {}
            ckt_latents = torch.randn(cnt, args['latent_dim']).to(args['device'])
            
            if scale_to_train_range:
                ckt_latents = ckt_latents * std + mean  # move to train's latent range

            batch['ckt_latents'] = ckt_latents
            for _ in range(decode_times):
                gen_graphs = model.archi.decode(args, batch)
                for g in gen_graphs:                    
                    if is_valid_DAG(g, start_symbol=(args['archiname']=='pace')):
                        num_valid_dags += 1
                        valid_dags.append(g)

                    if is_valid_Circuit(g, start_symbol=(args['archiname']=='pace')):
                        num_valid_ckts += 1
                        valid_ckts.append(g)
            cnt = 0

    time_end = time.time()
    comp_time = time_end - time_start

    rto_valid_ckt = num_valid_ckts / (n_latent_points * decode_times)
    rto_valid_dag = num_valid_dags / (n_latent_points * decode_times)
    rto_novel_dag = 1 - ratio_same_DAG(datasets['train'], valid_dags) # compare to the train datasets
    
    return {
        'infer_time': comp_time, 
        'rto_valid_ckt': rto_valid_ckt, 
        'rto_valid_dag': rto_valid_dag, 
        'rto_novel_dag': rto_novel_dag
    }
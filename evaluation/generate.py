"""Circuit generation from specifications.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import time
import torch
import numpy as np
from tqdm import tqdm

import utils.data as utils_data
import random
from scipy import linalg
from evaluation.tools import is_same_DAG, is_valid_Circuit, ratio_same_DAG, compute_retrieval_precision
from utils.data import print_graph

def calculate_spec_correct_ckt(gnd_specs, predict_specs):
    """Calculates specification accuracy for generated circuits.
    
    Compares predicted specifications against ground-truth specification ranges.
    A circuit is fully correct only if all three specs (gain, bw, pm) match.
    
    Args:
        gnd_specs: Dictionary with 'gains', 'bws', 'pms' keys, each containing
                  lists/tensors of valid specification values (ranges).
        predict_specs: Dictionary with 'gain', 'bw', 'pm' keys containing
                      predicted scalar values.
                      
    Returns:
        dict: Counts of correct predictions with keys:
            - 'total': Number of fully correct circuits (all 3 specs match)
            - 'gain': Number of correct gain predictions
            - 'bw': Number of correct bandwidth predictions
            - 'pm': Number of correct phase margin predictions
    """
    gnd_gains, gnd_bws, gnd_pms = gnd_specs['gains'], gnd_specs['bws'], gnd_specs['pms']
    total_right_nums = 0
    gain_right_nums = 0
    bw_right_nums = 0
    pm_right_nums = 0

    for i in range(len(gnd_gains)):
        pred_gain_i = predict_specs['gain'][i]
        pred_bw_i = predict_specs['bw'][i]
        pred_pm_i = predict_specs['pm'][i]

        if pred_gain_i not in gnd_gains[i]:
            continue
        else:
            gain_right_nums += 1

        if pred_bw_i not in gnd_bws[i]:
            continue
        else:
            bw_right_nums += 1

        if pred_pm_i not in gnd_pms[i]:
            continue
        else:
            pm_right_nums += 1
            
        total_right_nums += 1

    return {'total': total_right_nums, 'gain': gain_right_nums, 'bw': bw_right_nums, 'pm': pm_right_nums}


def calculate_cosine_distance(ckt_latents, spec_latents):
    """Calculates mean cosine distance between circuit and specification embeddings.
    
    Args:
        ckt_latents: Circuit embedding tensor of shape (batch_size, dim).
        spec_latents: Specification embedding tensor of shape (batch_size, dim).
        
    Returns:
        float: Mean cosine distance (1 - cosine_similarity).
    """
    cosine_sim = torch.nn.functional.cosine_similarity(ckt_latents, spec_latents, dim=-1)
    cosine_dist = 1 - cosine_sim
    return cosine_dist.mean()


def calculate_activation_statistics(activations):
    """Computes mean and covariance of activation vectors.
    
    Args:
        activations: Numpy array of shape (num_samples, feature_dim).
        
    Returns:
        tuple: (mean_vector, covariance_matrix) as numpy arrays.
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, times):
    """Calculates diversity as average pairwise L2 distance between random samples.
    
    Args:
        activation: Numpy array of shape (num_samples, feature_dim).
        times: Number of random pairs to sample.
        
    Returns:
        float: Mean L2 distance between random pairs.
        
    Raises:
        AssertionError: If activation is not 2D or has fewer samples than times.
    """
    assert len(activation.shape) == 2
    assert activation.shape[0] > times
    num_samples = activation.shape[0]
    
    first_indices = np.random.choice(num_samples, times, replace=False)
    second_indices = np.random.choice(num_samples, times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates Fréchet Inception Distance (FID) between two Gaussian distributions.
    
    FID measures the similarity between generated and real data distributions.
    Lower FID indicates more similar distributions.
    
    Args:
        mu1: Mean vector of first distribution.
        mu2: Mean vector of second distribution.
        sigma1: Covariance matrix of first distribution.
        sigma2: Covariance matrix of second distribution.
        eps: Small epsilon for numerical stability. Default 1e-6.
        
    Returns:
        float: Fréchet distance.
        
    Raises:
        ValueError: If imaginary component is too large (numerical instability).
        
    Notes:
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def evaluate_condition_generate(args, evaluator, datasets, logger, model):
    """Evaluates conditional circuit generation across multiple metrics.
    
    Generates circuits from specifications and evaluates:
    - Specification accuracy (gain, bw, pm)
    - Validity (DAG structure, circuit constraints)
    - Retrieval precision (R@1, R@2, R@3)
    - Diversity (intra-cluster and inter-cluster)
    - FID (Fréchet Inception Distance)
    - Figure of Merit (FoM)
    - Inference time
    
    Args:
        args: Configuration dictionary with model and evaluation settings.
        evaluator: Surrogate model for performance prediction.
        datasets: Dictionary with 'test' dataset.
        logger: Logger instance.
        model: Generative model.
        
    Returns:
        dict: Evaluation metrics with keys:
            - 'infer_time_gen': Total inference time
            - 'R_top1', 'R_top2', 'R_top3': Retrieval precision
            - 'spec_acc', 'gain_acc', 'bw_acc', 'pm_acc': Specification accuracy
            - 'cosine_distance': Mean multimodal distance
            - 'fid': Fréchet Inception Distance
            - 'intra_diversity': Diversity within samples
            - 'inter_diversity': Diversity across same spec
            - 'best_fom', 'avg_fom': Figure of merit statistics
            - 'rto_valid_ckt': Valid circuit ratio
    """
    sample_rounds = 120
    (
        total_num_spec_correct, 
        total_num_gain_correct,
        total_num_bw_correct,
        total_num_pm_correct,
        total_cosine_distance, 
        total_num_valid_ckts, 
        total_intra_diversity,
        total_inter_diversity,
        avg_fom,
        best_fom
    ) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    total_R_top1, total_R_top2, total_R_top3 = 0.0, 0.0, 0.0
    
    specs_clustered_ckts = utils_data.get_specification_domain(datasets['test'])

    logger.info('>>>>>>>>>>>>>>>>>>>>>>> Specification Numbers: %d'% (len(specs_clustered_ckts)))
    spec_numbers = len(specs_clustered_ckts)
    valid_ckts = []
    embs_gen = []
    embs_gnd = []
    infer_time = 0

    for rnd in tqdm(range(sample_rounds), desc="Evaluating Condition Generation", unit="Round"):
        ckts = []

        for i, cluster in enumerate(specs_clustered_ckts): # random select a ground-truth
            random_select_g = random.sample(cluster, 1)[0]
            ckts.append(random_select_g)

        _ckts = utils_data.collate_fn(ckts)
        batch = utils_data.transforms(args, _ckts)

        time_start = time.time()
        if args['modeltype']=='ldt':
            batch['ckt_latents'] = model(args, batch).squeeze(1)
            gen_ckts = model.vae.archi.decode(args, batch)
        elif args['modeltype']=='cvaegan':
            bsz = len(ckts)
            batch['ckt_latents'] = torch.randn(bsz, args['latent_dim']).to(args['device'])
            gen_ckts = model(args, batch)
        else:
            gen_ckts = model(args, batch) # generate circuits from specification

        time_end = time.time()
        this_infer_time = time_end - time_start
        infer_time += this_infer_time

        _gen_ckts = utils_data.collate_fn(gen_ckts)

        # if rnd == 0:
        ################################################################################################
        #                                        eval valid
        ################################################################################################
        num_valid_ckts = 0.0
        for _, g in enumerate(_gen_ckts):
            if is_valid_Circuit(g, start_symbol=(args['archiname']=='pace')):
                num_valid_ckts += 1
                valid_ckts.append(g)
        total_num_valid_ckts += num_valid_ckts

        if args['archiname'] == 'pace':
            # args['max_n'] -= 1
            for _, g in enumerate(_gen_ckts):
                utils_data.remove_start_symbol(g)

        batch_gen = utils_data.transform_digin(args, _gen_ckts)
        batch_gen = {key: val.to(args['device']) if torch.is_tensor(val) else val 
                                for key, val in batch_gen.items()}

        ckt_embs = evaluator.get_ckt_embeddings(batch_gen)
        spec_embs = evaluator.get_spec_embeddings(batch)

        cosine_distance = calculate_cosine_distance(ckt_embs, spec_embs)
        total_cosine_distance += cosine_distance

        ################################################################################################
        #                                    eval specification acc
        ################################################################################################
        predict_spec = evaluator.predict(ckt_embs)
        spec_acc_res = calculate_spec_correct_ckt(batch, predict_spec)
        best_fom     = max(best_fom, torch.max(predict_spec['fom']))
        avg_fom     += torch.sum(predict_spec['fom'])

        total_num_spec_correct += spec_acc_res['total']
        total_num_gain_correct += spec_acc_res['gain']
        total_num_bw_correct += spec_acc_res['bw']
        total_num_pm_correct += spec_acc_res['pm']

        ################################################################################################
        #                                         eval retrieval
        ################################################################################################
        gnds = list(range(len(ckts)))
        res_dict = compute_retrieval_precision(spec_embs, ckt_embs, gnds)
        total_R_top1 += res_dict['Top1']
        total_R_top2 += res_dict['Top2']
        total_R_top3 += res_dict['Top3']

        ################################################################################################
        #                                           eval fid
        ################################################################################################
        if args['archiname'] == 'pace':
            for _, g in enumerate(_ckts):
                utils_data.remove_start_symbol(g)

        batch_gnd = utils_data.transform_digin(args, _ckts)
        batch_gnd = {key: val.to(args['device']) if torch.is_tensor(val) else val 
                                    for key, val in batch_gnd.items()}
                                    
        ckt_embs_gnd = evaluator.get_ckt_embeddings(batch_gnd)
        
        ################################################################################################
        #                                       eval Intra
        ################################################################################################
        intra_diversity = calculate_diversity(ckt_embs.cpu().detach().numpy(), times=100)
        total_intra_diversity += intra_diversity
        
        embs_gen.append(ckt_embs.cpu().detach())
        embs_gnd.append(ckt_embs_gnd.cpu().detach())
        
        # logger.info('Round: %d, R@1: %.04f, R@2: %.04f, R@3: %.04f, MM-D: %.04f, Spec Acc: %.04f, Best FoM: %.04f, valid ckt: %.04f'% (
        #         rnd+1,
        #         res_dict['Top1'],
        #         res_dict['Top2'],
        #         res_dict['Top3'],
        #         cosine_distance,
        #         spec_acc_res['total'] / len(ckts),
        #         np.max(predict_spec['fom']),
        #         num_valid_ckts / len(ckts),
        #     )
        # )

    rto_valid_ckt = total_num_valid_ckts / (len(specs_clustered_ckts) * sample_rounds)

    ################################################################################################
    #                                       eval Inter
    ################################################################################################
    embs_gen_cluster = torch.stack(embs_gen, dim=0).cpu().numpy()
    _, spec_nums, _ = embs_gen_cluster.shape
    for i in range(spec_nums):
        same_spec_cluster = embs_gen_cluster[:, i, :]
        inter_diversity = calculate_diversity(same_spec_cluster, times=100)
        total_inter_diversity += inter_diversity


    R_top1 = total_R_top1 / sample_rounds
    R_top2 = total_R_top2 / sample_rounds
    R_top3 = total_R_top3 / sample_rounds

    spec_acc = total_num_spec_correct / (len(specs_clustered_ckts) * sample_rounds)
    gain_acc = total_num_gain_correct / (len(specs_clustered_ckts) * sample_rounds)
    bw_acc = total_num_bw_correct / (len(specs_clustered_ckts) * sample_rounds)
    pm_acc = total_num_pm_correct / (len(specs_clustered_ckts) * sample_rounds)
    avg_fom = avg_fom / (len(specs_clustered_ckts) * sample_rounds)

    ##### ---- eval frechet distance ---- #####
    mu_gnd_np = torch.cat(embs_gnd, dim=0).numpy()
    mu_gen_np = torch.cat(embs_gen, dim=0).numpy()
    mu_gnd, cov_gnd  = calculate_activation_statistics(mu_gnd_np)
    mu_gen, cov_gen  = calculate_activation_statistics(mu_gen_np)
    fid = calculate_frechet_distance(mu_gnd, cov_gnd, mu_gen, cov_gen)

    ##### ---- average metrics ---- #####
    cosine_distance = total_cosine_distance / sample_rounds
    intra_diversity = total_intra_diversity / sample_rounds
    inter_diversity = total_inter_diversity / spec_nums

    return {
        'avg_infer_time_per_ckt': infer_time * 1000 / (sample_rounds * spec_numbers),
        'R_top1': R_top1, 
        'R_top2': R_top2, 
        'R_top3': R_top3, 
        'spec_acc': spec_acc, 
        'gain_acc': gain_acc, 
        'bw_acc': bw_acc, 
        'pm_acc': pm_acc, 
        'cosine_distance': cosine_distance, 
        'fid': fid, 
        'intra_diversity': intra_diversity, 
        'inter_diversity': inter_diversity, 
        'best_fom': best_fom,
        'avg_fom': avg_fom,
        'rto_valid_ckt': rto_valid_ckt, 
    }


@torch.no_grad()
def generate_circuits(args, model, datasets, logger):
    """Generates circuits from specifications and logs/visualizes results.
    
    Generates circuits for each specification cluster, checks validity,
    and optionally plots the first 4 circuits.
    
    Args:
        args: Configuration dictionary with 'modelname', 'archiname'.
        model: Generative model.
        datasets: Dictionary with 'test' dataset.
        logger: Logger instance.
        
    Side Effects:
        - Logs circuit details and validity status
        - May create circuit visualization plots (first 4 circuits)
        
    Notes:
        Contains undefined variable 'num_valid_ckts' - appears to be incomplete code.
    """
    sample_rounds = 120
    specs_clustered_ckts = utils_data.get_specification_domain(datasets['test'])

    for rnd in tqdm(range(sample_rounds), desc="Generating circuits", unit="sample"):
        ckts = []
        
        for i, cluster in enumerate(specs_clustered_ckts):
            random_select_g = random.sample(cluster, 1)[0]
            ckts.append(random_select_g)

        _ckts = utils_data.collate_fn(ckts)
        batch = utils_data.transforms(args, _ckts)
        
        gen_ckts = model(args, batch) # generate according to the spec
        
        for _, g in enumerate(gen_ckts):

            logger.info('#########################' + args['modelname'] + '_' + str(i) + '#########################')
            logger.info(g)
            for i in range(g.vcount()):
                logger.info(g.vs[i].attributes())
                print(g.vs[i].attributes())
            if is_valid_Circuit(g, start_symbol=(args['archiname']=='pace')):

                num_valid_ckts += 1
                valid_ckts.append(g)

        for i in range(4):
            plot_graph(args, g, args['modelname'] + '_' + str(i), backbone=True, pdf=False)
    

def evaluate(args, model, datasets, logger):
    """Main evaluation entry point for conditional circuit generation.
    
    Loads a pretrained evaluator (surrogate model) and runs comprehensive
    generation evaluation including retrieval, specification accuracy,
    diversity, FID, and validity metrics.
    
    Args:
        args: Configuration with 'pretrained_eval_resume_pth' and 'device' keys.
        model: Generative model for circuit synthesis.
        datasets: Dictionary with 'test' dataset.
        logger: Logger instance.
        
    Side Effects:
        - Logs evaluation metrics summary
        - Evaluates model across multiple metrics and logs results
        
    Notes:
        Requires args['pretrained_eval_resume_pth'] to be set to a valid
        evaluator (surrogate) model checkpoint path.
    """
    model.eval()

    if 'pretrained_eval_resume_pth' not in args or args['pretrained_eval_resume_pth'] == '':
        logger.info('###############################################################################')
        logger.info('                  Need to load the pretrained model to evaluate')
        logger.info('###############################################################################')
    else:
        evaluator = torch.load(args['pretrained_eval_resume_pth'], map_location='cpu').to(args['device'])
        evaluator.eval()

        # evaluator_wrapper = EvaluatorForPerformancePrediction._from_pretrained(
        #     'Yuxuan-Hou/CktGen-Test',
        #     subfolder='evaluator-101'
        # )
        # evaluator = evaluator_wrapper.model.to(args['device'])
        # evaluator.eval()
        
        logger.info('###############################################################################')
        logger.info('                            Conditional Generation Evaluation')
        logger.info('###############################################################################')
        gen_res = evaluate_condition_generate(args, evaluator, datasets, logger, model)
        logger.info(
            'Avg Time/Ckt: %.4f ms, R@1: %.06f, R@2: %.06f, R@3: %.06f, Spec Acc: %.06f, MM-D: %.06f, FID: %.04f, Intra: %0.6f, Inter: %0.6f, Avg FoM %0.6f, Validity: %0.6f'%( 
                gen_res['avg_infer_time_per_ckt'],
                gen_res['R_top1'],
                gen_res['R_top2'],
                gen_res['R_top3'],
                gen_res['spec_acc'],
                gen_res['cosine_distance'],
                gen_res['fid'],
                gen_res['intra_diversity'],
                gen_res['inter_diversity'],
                gen_res['avg_fom'],
                gen_res['rto_valid_ckt']
            )
        )
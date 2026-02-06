"""Automated circuit design workflow.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import time
import torch
import pickle
import numpy as np
import utils.data as utils_data
import random
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
from tqdm import tqdm
from evaluation.tools import is_valid_Circuit

specification_domain = {
    'ckt_bench_101': {
        'gain': [0, 4],
        'bw': [0, 32],
        'pm': [0, 6]
    },
    'ckt_bench_301': {
        'gain': [0, 4],
        'bw': [0, 19],
        'pm': [0, 5]
    },
}

class Objective:
    """Optuna objective function for multi-armed bandit optimization in automated circuit design.
    
    This class implements a callable objective that samples specification candidates,
    generates circuits, evaluates them with a surrogate model, and tracks the best
    designs found during optimization.
    
    Attributes:
        args: Configuration dictionary.
        model: Generative model for circuit synthesis.
        surrogate: Surrogate model for performance prediction.
        constraint: Design constraints (gain, bw, pm) tuple.
        valid_candidates: List of valid specification tuples.
        optimize_sample_times: Number of circuits to generate per trial.
        best_fom: Best figure of merit found so far.
        best_spec: Specification that produced the best FoM.
        best_ckt: Best circuit found.
        total_fom: Cumulative FoM across all trials.
        total_correct_valid_nums: Count of valid circuits meeting constraints.
    """
    def __init__(self, args, model, surrogate, valid_candidates, constraint, optimize_sample_times):
        self.optimize_sample_times = optimize_sample_times
        self.args = args
        self.model = model
        self.surrogate = surrogate
        self.constraint = constraint

        self.valid_candidates = valid_candidates

        self.best_fom = -np.inf
        self.best_spec = None
        self.best_ckt = None

        self.total_fom = 0
        self.total_correct_valid_nums = 0


    @torch.no_grad()
    def __call__(self, trial):
        """Optuna trial evaluation function.
        
        Samples a specification candidate, generates circuits, predicts performance
        using surrogate model, and returns negative max FoM for minimization.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            float: Negative of maximum FoM (for minimization).
            
        Side Effects:
            Updates best_fom, best_spec, best_ckt, total_fom, total_correct_valid_nums.
        """
        idx = trial.suggest_categorical('candidate_idx', list(range(len(self.valid_candidates))))
        spec = self.valid_candidates[idx]

        max_fom = -np.inf
        best_ckt_local = None

        batch = {}
        batch['gains']   = torch.full((self.optimize_sample_times,), spec[0]).to(self.args['device'])
        batch['bws']     = torch.full((self.optimize_sample_times,), spec[1]).to(self.args['device'])
        batch['pms']     = torch.full((self.optimize_sample_times,), spec[2]).to(self.args['device'])
        
        gen_ckts = spec_cond_gen(self.args, self.model, batch)
        if self.args['archiname'] == 'pace':
            for _, g in enumerate(gen_ckts):
                utils_data.remove_start_symbol(g)
        preds = surrogate_model(self.args, self.surrogate, gen_ckts)

        foms = preds['fom']

        max_fom, max_idx = torch.max(foms, dim=0)
        best_ckt_local = gen_ckts[max_idx]

        for ckt, gain, bw, pm in zip(gen_ckts, preds['gain'], preds['bw'], preds['pm']):
            if is_specification_correct(self.constraint, gain, bw, pm) and is_valid_Circuit(ckt):
                self.total_correct_valid_nums += 1
        self.total_fom += torch.sum(foms, dim=0)

        if max_fom > self.best_fom:
            self.best_fom = max_fom
            self.best_spec = spec
            self.best_ckt = best_ckt_local

        return -max_fom


def get_all_specification_candidates(datasets):
    """Extracts unique specifications from train and test datasets.
    
    Floors each specification (gain, bw, pm) to nearest integer and
    collects unique combinations.
    
    Args:
        datasets: Dictionary with 'train' and 'test' keys containing graph lists.
        
    Returns:
        set: Set of (gain, bw, pm) tuples representing unique specifications.
    """
    unique_specs_set = {
        (
            utils_data.floor_to_decimal(g['gain'], 0),
            utils_data.floor_to_decimal(g['bw'], 0),
            utils_data.floor_to_decimal(g['pm'], 0)
        )
        for g in datasets['train'] + datasets['test']
    }

    return unique_specs_set


def get_valid_candidates(candidates, constraint):
    """Filters specification candidates that meet given constraints.
    
    Args:
        candidates: Iterable of (gain, bw, pm) tuples.
        constraint: (gain_min, bw_min, pm_min) constraint tuple.
        
    Returns:
        list: Filtered list of (gain, bw, pm) tuples meeting all constraints.
    """
    gain_cons, bw_cons, pm_cons = constraint
    return [(gain, bw, pm) for gain, bw, pm in candidates if gain >= gain_cons and bw >= bw_cons and pm >= pm_cons]

def is_specification_correct(constraint, gain, bw, pm):
    """Checks if a specification meets design constraints.
    
    Args:
        constraint: (gain_min, bw_min, pm_min) constraint tuple.
        gain: Gain value (can be tensor or scalar).
        bw: Bandwidth value (can be tensor or scalar).
        pm: Phase margin value (can be tensor or scalar).
        
    Returns:
        bool: True if all constraints are satisfied.
    """
    if gain < constraint[0]:
        return False
    if bw < constraint[1]:
        return False
    if pm < constraint[2]:
        return False
    return True


def get_legal_random_outer_constraints(args, cons_random_sample_times, constraints, candidates):
    """Generates random constraints outside existing ones for out-of-distribution testing.
    
    Creates new constraint combinations from the specification domain that:
    1. Are not in the existing constraint set
    2. Have at least one valid candidate satisfying them
    
    Args:
        args: Configuration with 'data_name' key.
        cons_random_sample_times: Maximum number of random constraints to sample.
        constraints: Existing constraint set to exclude.
        candidates: Available specification candidates.
        
    Returns:
        set: Set of randomly sampled (gain, bw, pm) constraint tuples.
    """
    gain_range = range(specification_domain[args['data_name']]['gain'][0], specification_domain[args['data_name']]['gain'][1])
    bw_range   = range(specification_domain[args['data_name']]['bw'][0], specification_domain[args['data_name']]['bw'][1])
    pm_range   = range(specification_domain[args['data_name']]['pm'][0], specification_domain[args['data_name']]['pm'][1])
    all_possible_cons = set((g, b, p) for g in gain_range for b in bw_range for p in pm_range)
    
    available_cons = all_possible_cons - set(constraints)

    filtered_cons = [cons for cons in available_cons if len(get_valid_candidates(candidates, cons)) > 0]
    
    sample_num = min(cons_random_sample_times, len(filtered_cons))
    print('sample num: ', sample_num)
    random_constraints = set(random.sample(filtered_cons, sample_num))
    return random_constraints

# --------------------------------------------------
# 2. 模拟图生成器和代理模型
# --------------------------------------------------
@torch.no_grad()
def spec_cond_gen(args, model, batch):
    """Generates circuits conditioned on specifications using different model types.
    
    Supports three model architectures:
    - ldt: Latent diffusion transformer
    - cvaegan: Conditional VAE-GAN (with random latent sampling)
    - cktgen: Direct conditional generation
    
    Args:
        args: Configuration with 'modeltype', 'latent_dim', 'device' keys.
        model: Generative model.
        batch: Batch dictionary with 'gains', 'bws', 'pms' specification tensors.
        
    Returns:
        list: Generated circuit graphs (igraph.Graph objects).
    """
    if args['modeltype']=='ldt':
        batch['ckt_latents'] = model(args, batch).squeeze(1)
        gen_ckts = model.vae.archi.decode(args, batch)
    elif args['modeltype']=='cvaegan':
        bsz = len(batch['gains'])
        batch['ckt_latents'] = torch.randn(bsz, args['latent_dim']).to(args['device'])
        gen_ckts = model(args, batch)
    else: # cktgen
        gen_ckts = model(args, batch) # generate circuits from specification
    return gen_ckts

@torch.no_grad()
def surrogate_model(args, surrogate, circuits):
    """Predicts circuit performance using a surrogate model.
    
    Args:
        args: Configuration dictionary.
        surrogate: Surrogate model with get_ckt_embeddings and predict methods.
        circuits: List of circuit graphs (igraph.Graph objects).
        
    Returns:
        dict: Predicted specifications with 'fom', 'gain', 'bw', 'pm' keys.
    """
    batch = utils_data.transform_digin(args, circuits)
    batch = {key: val.to(args['device']) if torch.is_tensor(val) else val 
                            for key, val in batch.items()}
    ckt_embs = surrogate.get_ckt_embeddings(batch)
    predict_spec = surrogate.predict(ckt_embs)
    return predict_spec


def auto_design(args, model, surrogate, datasets, logger):
    """Performs automated circuit design using multi-armed bandit optimization.

    For each constraint:
    1. Identifies valid specification candidates
    2. Uses TPE (Tree-structured Parzen Estimator) sampler for Bayesian optimization
    3. Generates circuits and evaluates with surrogate model
    4. Tracks best designs and aggregate metrics
    
    Args:
        args: Configuration dictionary with 'device', 'archiname', 'out_dir', 'modeltype'.
        model: Generative model for circuit synthesis.
        surrogate: Surrogate model for performance prediction.
        datasets: Dictionary with 'train' and 'test' datasets.
        logger: Logger instance.
        
    Side Effects:
        - Logs optimization progress and results
        - Saves results to {out_dir}/{modeltype}.pkl with keys:
          'acc', 'fom', 'best_acc', 'best_fom'
    """
    # --------------------------------------------------
    # 1. 初始化候选空间和约束
    # --------------------------------------------------
    candidates = get_all_specification_candidates(datasets)
    cons_random_sample_times = 50
    optimize_sample_times = 10

    total_best_correct_valid_nums = 0
    total_best_fom = 0

    total_avg_search_acc = []
    total_avg_search_fom = []
    best_acc = []
    best_fom = []
    
    constraints = candidates
    random_constraints = get_legal_random_outer_constraints(args, cons_random_sample_times, constraints, candidates)
    constraints.update(random_constraints)

    logger.info('>>>>>>>>>>>>>>>>>>>>>>> Total Constraints Numbers: %d'% (len(constraints)))
    logger.info('Processing constraints with Optuna optimization...')
    for i, cons in enumerate(tqdm(constraints, desc="Auto design optimization", unit="constraint")):
        # random select constraints
        valid_candidates = get_valid_candidates(candidates, cons)
        
        ####################################################################################
        # Perform Multi-Arm-Bandit Optimization
        ####################################################################################
        objective = Objective(
            args=args, 
            model=model, 
            surrogate=surrogate, 
            valid_candidates=valid_candidates, 
            constraint=cons,
            optimize_sample_times=optimize_sample_times
        )
        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=len(valid_candidates)*2)

        searched_avg_fom = objective.total_fom / (len(valid_candidates)*2 * optimize_sample_times)
        searched_avg_acc = objective.total_correct_valid_nums / (len(valid_candidates)*2 * optimize_sample_times)

        ####--- eval best ---####
        pred = surrogate_model(args, surrogate, [objective.best_ckt])
        is_correct = is_specification_correct(cons, pred['gain'], pred['bw'], pred['pm'])
        is_valid = is_valid_Circuit(objective.best_ckt, start_symbol=(args['archiname']=='pace'))

        if is_correct and is_valid:
            total_best_correct_valid_nums += 1
            best_acc.append(1)
        else:
            best_acc.append(0)
        total_best_fom += objective.best_fom
        best_fom.append(objective.best_fom)


        logger.info('####################### %d th constraint #######################'%(i + 1)) 
        logger.info('Given constraints, Gain > %d, BW > %d, PM > %d'%(cons[0], cons[1], cons[2])) 
        logger.info('Best searched FoM: %.06f, Correct?: %s, Valid?: %s;  Avg Searched Acc: %.06f, Avg searched FoM: %.06f'%(
            objective.best_fom,
            is_correct, 
            is_valid,
            searched_avg_acc,
            searched_avg_fom,
        )) 

        total_avg_search_acc.append(searched_avg_acc)
        total_avg_search_fom.append(searched_avg_fom)
        
    logger.info('Correct and Valid Design Ratrio: %.06f, Avg Best FoM: %.06f'%(total_best_correct_valid_nums / len(constraints), total_best_fom / len(constraints)))

    from pathlib import Path
    filename = Path(args['out_dir']) / f"{args['modeltype']}.pkl"
    save = {
        'acc': total_avg_search_acc, 
        'fom': total_avg_search_fom,
        'best_acc': best_acc,
        'best_fom': best_fom
    }
    with open(filename, 'wb') as f:
        pickle.dump(save, f)

def evaluate(args, model, datasets, logger):
    """Main evaluation entry point for automated inverse circuit design.
    
    Loads a pretrained surrogate model and runs inverse design evaluation
    across multiple design constraints.
    
    Args:
        args: Configuration with 'pretrained_eval_resume_pth' and 'device' keys.
        model: Generative model for circuit synthesis.
        datasets: Dictionary with 'train' and 'test' datasets.
        logger: Logger instance.
        
    Notes:
        Requires args['pretrained_eval_resume_pth'] to be set to a valid
        surrogate model checkpoint path.
    """
    if 'pretrained_eval_resume_pth' not in args or args['pretrained_eval_resume_pth'] == '':
        logger.info('###############################################################################')
        logger.info('                  Need to load the pretrained model to evaluate')
        logger.info('###############################################################################')
    else:
        logger.info('###############################################################################')
        logger.info('                            Auto Design Evaluation')
        logger.info('###############################################################################')

        model.eval()
        surrogate = torch.load(args['pretrained_eval_resume_pth'], map_location='cpu').to(args['device'])
        # fom_predictor_dir = os.path.dirname(args['pretrained_eval_resume_pth'])
        # evaluator.fom_predictor.load(fom_predictor_dir)
        surrogate.eval()

        auto_design(args, model, surrogate, datasets, logger)
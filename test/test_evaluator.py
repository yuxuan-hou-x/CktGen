"""Evaluation entry point for trained evaluator models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import os
import json
import torch
import random
import numpy as np

import utils.logger as utils_logger
import utils.paths as utils_paths

from utils.checkpoint import load_model_checkpoint
from options.training import parser
from dataset.get_datasets import get_datasets

import evaluation.prediction as eval_prediction


def main():
    """Load a trained evaluator checkpoint and run evaluator evaluation."""
    args = parser()

    seed = args['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    utils_paths.setup_paths(args)
    os.makedirs(args['out_dir'], exist_ok=True)

    logger = utils_logger.get_logger(args['out_dir'], args.get('exp_name'))
    logger.info(json.dumps(args, indent=4, sort_keys=True, default=utils_logger.serialize))

    datasets = get_datasets(args)

    logger.info(
        'Training data length: %d, Test data length: %d'
        % (len(datasets['train']), len(datasets['test']))
    )

    if 'resume_pth' not in args or args['resume_pth'] is None:
        raise ValueError('resume_pth is required for evaluator evaluation')

    logger.info('loading checkpoint from {}'.format(args['resume_pth']))

    model = load_model_checkpoint(args['resume_pth'], map_location='cpu').to(args['device'])
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Number of model parameters: %d' % total_params)

    eval_prediction.evaluate(args, model, datasets, logger)


if __name__ == "__main__":
    main()

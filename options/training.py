"""Training hyperparameter configuration options.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


from options.base import add_misc_options, adding_cuda, add_cuda_options, ArgumentParser
from options.dataset import add_dataset_options
from options.evaluation import add_evaluate_options
from options.models import parse_modelname
import options.models as models_options

def add_training_options(parser):
    """Add training hyperparameter options.
    
    These options control the optimization process including learning rate, batch size, and checkpointing.
    """
    group = parser.add_argument_group('Training options')
    group.add_argument(
        '--lr', 
        type=float, 
        default=1e-4, 
        metavar='LR', 
        help='Learning rate for the AdamW optimizer. Typical values are 1e-4 to 1e-3. '
             'The learning rate will be automatically reduced by a factor of 0.1 if training loss plateaus '
             '(controlled by ReduceLROnPlateau scheduler with patience=10).'
    )
    group.add_argument(
        '--epochs', 
        type=int, 
        default=100000, 
        metavar='N', 
        help='Maximum number of training epochs. Training will run for this many epochs unless stopped early. '
             'Each epoch iterates through the entire training dataset once. '
             'Note: The default value of 100000 is intentionally high; training is typically stopped earlier using checkpoints.'
    )
    group.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        metavar='N', 
        help='Number of circuit graphs in each training batch. Larger batch sizes provide more stable gradients '
             'but require more GPU memory. Typical values: 16-64 depending on model size and available GPU memory.'
    )
    group.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        metavar='S', 
        help='Random seed for reproducibility. Sets the seed for PyTorch, NumPy, and Python random module. '
             'Using the same seed ensures identical results across runs (assuming same hardware/software).'
    )
    group.add_argument(
        '--beta', 
        default=[0.9, 0.99], 
        nargs='+', 
        type=float, 
        help='Beta coefficients (beta1, beta2) for the AdamW optimizer. Controls exponential moving averages '
             'of gradient and squared gradient. Default [0.9, 0.99] works well for most cases. '
             'Format: --beta 0.9 0.99'
    )
    group.add_argument(
        '--weight_decay', 
        default=0.0, 
        type=float, 
        help='L2 regularization coefficient for AdamW optimizer. Helps prevent overfitting by penalizing large weights. '
             'Default 0.0 means no regularization. Typical non-zero values: 1e-4 to 1e-2.'
    )
    group.add_argument(
        '--save_interval', 
        type=int, 
        default=100, 
        metavar='N', 
        help='Checkpoint saving frequency in epochs. Model checkpoints will be saved every N epochs. '
             'Checkpoints are saved to {out_dir}/{modelname}_checkpoint{epoch}.pth. '
             'Use smaller values (e.g., 50) for more frequent checkpoints or larger (e.g., 200) to save disk space.'
    )

def parser():
    """Builds and parses all training configuration arguments.
    
    This is the main configuration parser for training. It assembles arguments from multiple
    option groups (misc, CUDA, training, evaluation, dataset, model), parses the model name
    to determine architecture-specific options, and returns a unified configuration dictionary.
    
    The parser uses a two-stage parsing strategy:
    1. First pass: Parse base options and model name
    2. Second pass: Parse architecture-specific options based on the model name
    
    Returns:
        dict: Complete configuration dictionary containing all parsed arguments with keys including:
            - Basic settings: 'out_dir', 'note', 'no_cuda', 'which_cuda', 'seed'
            - Training params: 'lr', 'epochs', 'batch_size', 'beta', 'weight_decay', 'save_interval'
            - Evaluation params: 'test_batch_size', 'start_epoch', 'ckpt'
            - Dataset params: 'dataset', 'data_dir', 'split_ratio', 'num_workers'
            - Model params: 'modelname', 'modeltype', 'archiname', 'losses', 'lambdas'
            - Conditional params: 'conditioned', 'vae', 'contrastive', 'guided', 'filter'
            - Loss weights: 'lambda_recon', 'lambda_kl', 'lambda_align', 'lambda_nce', 'lambda_gde', 'lambda_pred'
            - Architecture-specific params (e.g., for CktArchi):
                'max_n', 'emb_dim', 'hidden_dim', 'latent_dim', 'num_layers', 'num_heads', etc.
            - CUDA settings: 'device' (torch.device object)
            
    Notes:
        - The 'lambdas' key contains a dictionary mapping loss names to their weights.
        - Architecture-specific options are dynamically added based on 'archiname'.
        - Only non-None arguments are included in the returned dictionary.
        - CUDA device is automatically configured based on availability and user preferences.
        
    Examples:
        >>> args = parser()
        >>> print(args['modelname'])
        'cktgen_cktarchi_kl_recon_align_nce_gde'
        >>> print(args['modeltype'], args['archiname'])
        'cktgen' 'cktarchi'
        >>> print(args['lambdas'])
        {'kl': 0.005, 'recon': 1.0, 'align': 1.0, 'nce': 1.0, 'gde': 1.0}
    """
    parser = ArgumentParser()

    # misc options
    add_misc_options(parser)
    
    # cuda options
    add_cuda_options(parser)
    
    # training options
    add_training_options(parser)
    
    # evaluation options
    add_evaluate_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    models_options.add_model_options(parser)

    # model architecture options
    opt, rest = parser.parse_known_args()

    # remove None params, and create a dictionnary
    args = {key: val for key, val in vars(opt).items() if val is not None}

    # parse modelname
    args['modeltype'], args['archiname'], args['losses'] = parse_modelname(args['modelname'])
    lambdas = {}
    for loss in args['losses']:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    args["lambdas"] = lambdas

    # parse architect args
    parser_archi = ArgumentParser() 
    add_archi_args_options = models_options.__getattribute__(f"add_{args['archiname']}_options")
    add_archi_args_options(parser_archi)
    opt_archi, _ = parser_archi.parse_known_args(rest)

    args_archi = {key: val for key, val in vars(opt_archi).items() if val is not None}
    args.update(args_archi)

    adding_cuda(args)
    
    return args
"""Model architecture and training configuration options.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


from models.get_model import MODELTYPES, ARCHINAMES, LOSSES


def add_cktgnn_options(parser):
    """
    Add CKTGNN (Circuit Graph Neural Network) architecture options.
    CKTGNN uses GRU-based sequential encoding/decoding with graph structure.
    This is one of the baseline architectures for circuit generation.
    """
    group = parser.add_argument_group('CKTGNN model options')

    group.add_argument('--max_n', type=int, default=8, help='')
    group.add_argument('--max_pos', type=int, default=8, help='')
    group.add_argument('--num_types', type=int, default=26, help='number of different node (subgraph) types')
    group.add_argument('--emb_dim', type=int, default=24, help='embdedding dimension')
    group.add_argument('--feat_emb_dim', type=int, default=8, help='embedding dimension of subg feats')
    group.add_argument('--hid_dim', type=int, default=301, help='hidden size of GRUs')
    group.add_argument('--latent_dim', type=int, default=66, help='embedding dimension of latent space')
    group.add_argument('--bidirectional', action='store_true', default=False, help='whether to use bidirectional encoding')
    group.add_argument('--sized', action='store_true', default=True, help='whether to use bidirectional encoding')


def add_cktarchi_options(parser):
    """
    Add CktArchi (Circuit Transformer Architecture) model options.
    CktArchi uses transformer-based encoder-decoder architecture with graph-aware embeddings.
    This is the main architecture used in CktGen for high-quality circuit generation.
    """
    group = parser.add_argument_group('CKT Transformer model options')

    group.add_argument(
        '--max_n', 
        type=int, 
        default=8, 
        help='Maximum number of nodes (devices) in a circuit. Defines the maximum sequence length for transformers. '
             'For CktBench101/301: max_n=8 (excluding START symbol) or max_n=9 (including START for PACE architecture).'
    )
    group.add_argument(
        '--emb_dim', 
        type=int, 
        default=128, 
        help='Base embedding dimension for node types, paths, and topology. All discrete features are embedded to this dimension '
             'before being concatenated and projected to hidden_dim.'
    )
    group.add_argument(
        '--hidden_dim', 
        type=int, 
        default=512, 
        help='Hidden dimension of the transformer layers. This is the d_model parameter in standard transformers. '
             'All transformer computations (self-attention, cross-attention, feedforward) use this dimension.'
    )
    group.add_argument(
        '--latent_dim', 
        type=int, 
        default=64, 
        help='Dimension of the latent representation z in VAE or the conditional embedding space. '
             'For VAE: mu and logvar are projected to this dimension. For conditional models: specs are embedded to this dimension.'
    )
    group.add_argument(
        '--size_emb_dim', 
        type=int, 
        default=8, 
        help='Embedding dimension for continuous size features (R, C, gm values of devices). '
             'These continuous features are embedded separately and concatenated with discrete embeddings.'
    )
    group.add_argument(
        '--ff_size', 
        type=int, 
        default=512, 
        help='Size of the feedforward network (FFN) hidden layer in transformer blocks. '
             'Standard transformers use 4*hidden_dim, but this can be customized. Larger values increase model capacity.'
    )
    group.add_argument(
        '--num_layers', 
        type=int, 
        default=4, 
        help='Number of transformer encoder/decoder layers. More layers can capture more complex patterns '
             'but increase training time and risk overfitting. Typical range: 3-6 layers.'
    )
    group.add_argument(
        '--num_heads', 
        type=int, 
        default=8, 
        help='Number of attention heads in multi-head attention. hidden_dim must be divisible by num_heads. '
             'More heads allow the model to attend to different aspects of the input simultaneously.'
    )
    group.add_argument(
        '--num_types', 
        type=int, 
        default=26, 
        help='Total number of device types in the circuit vocabulary (transistors, resistors, capacitors, etc.). '
             'Determines the size of type embedding layer. CktBench uses 26 types.'
    )
    group.add_argument(
        '--num_paths', 
        type=int, 
        default=8, 
        help='Number of different path positions in the circuit topology. '
             'Each device is assigned a path index indicating its role in the circuit (e.g., input stage, output stage). '
             'Determines the size of path embedding layer.'
    )
    group.add_argument(
        '--block_size', 
        type=int, 
        default=9, 
        help='Maximum sequence length for autoregressive generation in the decoder. '
             'Set to max_n + 1 to account for START token. Used in GPT-style decoder architectures.'
    )
    group.add_argument(
        '--dropout_rate', 
        type=float, 
        default=0.3, 
        help='Dropout probability applied in transformer layers (attention and feedforward). '
             'Helps prevent overfitting. Typical range: 0.1-0.3. Higher values for smaller datasets.'
    )
    group.add_argument(
        '--fc_rate', 
        type=int, 
        default=4, 
        help='Multiplier for feedforward network hidden size relative to hidden_dim. '
             'Standard transformer uses fc_rate=4, meaning FFN hidden size = 4 * hidden_dim. '
             'Not used if ff_size is explicitly set.'
    )
    group.add_argument(
        '--type_rate', 
        type=float, 
        default=0.5, 
        help='Weight for type prediction loss in the multi-task loss function. '
             'Controls the importance of correctly predicting device types during generation. '
             'Total reconstruction loss = type_rate * type_loss + path_rate * path_loss + size_rate * size_loss + edge_loss.'
    )
    group.add_argument(
        '--path_rate', 
        type=float, 
        default=0.05, 
        help='Weight for path position prediction loss in the multi-task loss function. '
             'Controls the importance of correctly predicting device path positions.'
    )
    group.add_argument(
        '--size_rate', 
        type=float, 
        default=0.01, 
        help='Weight for continuous size feature (R, C, gm) prediction loss in the multi-task loss function. '
             'Lower than type_rate because continuous values are less critical than discrete topology.'
    )


def add_digin_options(parser):
    """
    Add DIGIN (Device-level Iterative Graph Inference Network) model options.
    DIGIN is a GIN-based architecture for circuit encoding/decoding.
    Similar to CktArchi but uses Graph Isomorphism Network instead of transformers.
    """
    group = parser.add_argument_group('DIGIN model options')

    group.add_argument(
        '--max_n', 
        type=int, 
        default=8, 
        help='Maximum number of nodes (devices) in a circuit. Defines graph size limits.'
    )
    group.add_argument(
        '--emb_dim', 
        type=int, 
        default=128, 
        help='Base embedding dimension for node types, paths, and topology features.'
    )
    group.add_argument(
        '--hidden_dim', 
        type=int, 
        default=512, 
        help='Hidden dimension for graph neural network layers and MLP projections.'
    )
    group.add_argument(
        '--latent_dim', 
        type=int, 
        default=64, 
        help='Dimension of the latent representation z for VAE or conditional embeddings.'
    )
    group.add_argument(
        '--size_emb_dim', 
        type=int, 
        default=8, 
        help='Embedding dimension for continuous device size features (R, C, gm).'
    )
    group.add_argument(
        '--num_types', 
        type=int, 
        default=26, 
        help='Number of device types in the circuit vocabulary. Determines type embedding size.'
    )
    group.add_argument(
        '--num_paths', 
        type=int, 
        default=8, 
        help='Number of path position types. Determines path embedding size.'
    )
    group.add_argument(
        '--dropout', 
        type=float, 
        default=0.1, 
        help='Dropout probability in GIN layers. Lower than transformer dropout due to simpler architecture.'
    )


def add_pace_options(parser):
    """
    Add PACE (Positional Autoregressive Circuit Encoder) model options.
    
    PACE is a transformer-based autoregressive model with explicit START symbol handling.
    It's a baseline architecture from prior work, adapted for circuit generation.
    """
    group = parser.add_argument_group('PACE model options')
    group.add_argument('--num_types', type=int, default=27, help='number of different node (subgraph) types')
    group.add_argument('--emb_dim', type=int, default=32, help='position embedding and embedding size')
    group.add_argument('--v_size_emb_dim', type=int, default=8, help='node feature embedding size')
    group.add_argument('--hidden_dim', type=int, default=96, help='dimension of hidden state of transformer')
    group.add_argument('--num_heads', type=int, default=8, help='number of heads in self attention')
    group.add_argument('--num_layers', type=int, default=3, help='number of self attention layers')
    group.add_argument('--dropout', type=float, default=0.15, help='dropout rate in transformer')
    group.add_argument('--fc_hidden', type=int, default=32, help='')
    group.add_argument('--latent_dim', type=int, default=96, help='number of dimensions of latent vectors z')


def add_model_options(parser):
    """
    Add general model configuration and loss function options.
    These options control the model type, architecture choice, training objectives, and loss weights.
    """
    group = parser.add_argument_group('model options')
    
    group.add_argument(
        '--modelname', 
        default='cktgen_cktarchi_kl_recon_align_nce_gde', 
        help='Model identifier string specifying model type, architecture, and losses. '
             'Format: {modeltype}_{archiname}_{loss1}_{loss2}_... '
             'Model types: cktgen (conditional gen), vae (reconstruction), ldt (latent diffusion), cvaegan (GAN-based), evaluator '
             'Architectures: cktarchi (transformer), pace, cktgnn (GRU), digin (GIN) '
             'Losses: recon, kl, align, nce, gde, pred, mse, nll '
             'Examples: "cktgen_cktarchi_kl_recon_align_nce_gde", "vae_cktarchi_kl_recon", "ldt_cktarchi"'
    )
    
    group.add_argument(
        '--conditioned', 
        action='store_true', 
        default=False, 
        help='Enable conditional generation based on circuit specifications (gain, bandwidth, phase margin). '
             'When True, the model learns to generate circuits that meet specified performance requirements. '
             'Automatically set to True for cktgen, ldt, cvaegan, and evaluator model types.'
    )
    
    group.add_argument(
        '--vae', 
        action='store_true', 
        default=False, 
        help='Use variational autoencoder (VAE) with KL divergence regularization. '
             'When True, the encoder outputs mu and logvar for sampling latent z via reparameterization trick. '
             'When False, the encoder directly outputs a deterministic latent vector. '
             'Required for proper probabilistic generation and diversity.'
    )
    
    group.add_argument(
        '--eps_factor', 
        type=float, 
        default=1, 
        help='Scaling factor for the epsilon noise in VAE reparameterization trick: z = mu + eps_factor * std * epsilon. '
             'Default 1.0 uses standard VAE. Values < 1.0 reduce stochasticity. '
             'Used in CKTGNN and PACE architectures; CktArchi uses standard reparameterization.'
    )
    
    group.add_argument(
        '--contrastive', 
        action='store_true', 
        default=False, 
        help='Enable contrastive learning via InfoNCE loss to align circuit and specification embeddings. '
             'Learns to pull together circuits with the same specs and push apart circuits with different specs. '
             'Improves spec-to-circuit generation quality. Requires --lambda_nce to be set.'
    )
    
    group.add_argument(
        '--guided', 
        action='store_true', 
        default=False, 
        help='Enable classifier-free guidance during generation. The model learns both conditional and unconditional generation, '
             'then uses guidance to strengthen the conditioning signal at inference time. '
             'Improves adherence to specifications. Requires --lambda_gde to be set.'
    )
    
    group.add_argument(
        '--filter', 
        action='store_true', 
        default=False, 
        help='Use filtered contrastive learning that masks out false negatives in InfoNCE loss. '
             'In a batch, circuits with the same specs are not treated as negatives to each other. '
             'Improves training stability when multiple circuits share the same specification. '
             'Only effective when --contrastive is also enabled.'
    )
    
    # Loss weights
    group.add_argument(
        '--lambda_recon', 
        type=float, 
        default=1, 
        help='Weight for reconstruction loss (cross-entropy for discrete predictions + MSE for continuous features). '
             'Controls how accurately the model reconstructs circuit topology and device parameters. '
             'Default 1.0. Increase if reconstruction quality is poor.'
    )
    
    group.add_argument(
        '--lambda_kl', 
        type=float, 
        default=5e-3, 
        help='Weight for KL divergence loss in VAE: KL(q(z|x) || p(z)). Regularizes the latent space to be close to N(0,I). '
             'Typical values: 1e-5 to 1e-2. Lower values give better reconstruction but less smooth latent space. '
             'Higher values enforce stronger regularization but may blur generation. Common: 5e-3, 1e-3, 1e-5.'
    )
    
    group.add_argument(
        '--lambda_align', 
        type=float, 
        default=1, 
        help='Weight for cross-modal alignment loss (MSE between circuit latents and spec latents). '
             'Encourages the latent spaces of circuits and specifications to be aligned. '
             'Essential for conditional generation. Default 1.0.'
    )
    
    group.add_argument(
        '--lambda_gde', 
        type=float, 
        default=1, 
        help='Weight for classifier-free guidance loss. Only used when --guided is True. '
             'Controls the strength of the guidance signal. Default 1.0.'
    )
    
    group.add_argument(
        '--lambda_nce', 
        type=float, 
        default=1, 
        help='Weight for InfoNCE contrastive loss. Only used when --contrastive is True. '
             'Pulls together (circuit, spec) pairs and pushes apart mismatched pairs. '
             'Typical values: 0.1 to 1.0. Default 1.0.'
    )
    
    group.add_argument(
        '--lambda_pred', 
        type=float, 
        default=1, 
        help='Weight for FoM prediction loss in the evaluator model. '
             'Used only in training the evaluator, not in cktgen training. '
             'Controls the trade-off between spec prediction accuracy and embedding quality.'
    )
    
    group.add_argument(
        '--lambda_mse', 
        type=float, 
        default=1, 
        help='Weight for mean squared error loss in the evaluator model. '
    )

    group.add_argument(
        '--temperature', 
        type=float, 
        default=0.1, 
        help='Temperature parameter for InfoNCE contrastive loss: exp(sim(a,b)/T). '
             'Lower temperatures (0.05-0.1) make the model more discriminative but harder to train. '
             'Higher temperatures (0.2-0.5) make training easier but less precise. '
             'Typical range: 0.07 to 0.2. Default 0.1.'
    )


def parse_modelname(modelname):
    """Parses model name string into model type, architecture, and loss components.
    
    The model name follows the format: {modeltype}_{archiname}_{loss1}_{loss2}_...
    This function validates each component and returns them separately.
    
    Args:
        modelname: Model identifier string, e.g., "cktgen_cktarchi_kl_recon_align_nce_gde".
        
    Returns:
        tuple: (modeltype, archiname, losses) where:
            - modeltype (str): Model type ('cktgen', 'vae', 'ldt', 'cvaegan', 'evaluator').
            - archiname (str): Architecture name ('cktarchi', 'pace', 'cktgnn', 'digin').
            - losses (list): List of loss function names (e.g., ['kl', 'recon', 'align', 'nce', 'gde']).
            
    Raises:
        NotImplementedError: If modeltype, archiname, or any loss is not in the supported lists,
                           or if no loss functions are specified.
                           
    Examples:
        >>> parse_modelname('cktgen_cktarchi_kl_recon_align')
        ('cktgen', 'cktarchi', ['kl', 'recon', 'align'])
        
        >>> parse_modelname('vae_digin_kl_recon')
        ('vae', 'digin', ['kl', 'recon'])
    """
    modeltype, archiname, *losses = modelname.split('_')

    if modeltype not in MODELTYPES:
        raise NotImplementedError('This type of model is not implemented.')
    if archiname not in ARCHINAMES:
        raise NotImplementedError('This architechture is not implemented.')

    if len(losses) == 0:
        raise NotImplementedError("You have to specify at least one loss function.")
    
    for loss in losses:
        if loss not in LOSSES:
            raise NotImplementedError("This loss is not implemented.")

    return modeltype, archiname, losses

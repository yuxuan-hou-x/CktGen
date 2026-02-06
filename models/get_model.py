"""Model factory for creating model instances.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module provides a factory function for instantiating models based on
configuration parameters. It dynamically loads and combines model types
(VAE, CktGen, LDT, etc.) with architectures (CktArchi, PACE, etc.).

Supported model types:
    - vae: Variational Autoencoder
    - cktgen: Conditional VAE with specifications
    - ldt: Latent Diffusion Transformer
    - cvaegan: Conditional VAE-GAN
    - evaluator: Performance predictor

Supported architectures:
    - cktarchi: Transformer-based encoder-decoder
    - pace: Parallel convolution encoder
    - cktgnn: Graph Neural Network
    - digin: Lightweight GNN

Example:
    >>> args = {'modeltype': 'cktgen', 'archiname': 'cktarchi', ...}
    >>> model = get_model(args)
"""


import importlib

# Registry of available model types, architectures, and losses
MODELTYPES = ['cktgen', 'vae', 'ldt', 'cvaegan', 'evaluator']
ARCHINAMES = ['cktarchi', 'pace', 'cktgnn', 'digin']
LOSSES = ["recon", "kl", "align", "nce", "gde", 'pred', 'mse', 'nll']


def get_model(args):
    """Factory function to create a model based on configuration.
    
    Dynamically imports and instantiates the specified model type and
    architecture, combining them according to the model's requirements.
    
    Args:
        args: Dictionary containing model configuration with keys:
            - 'modeltype': Type of model ('vae', 'cktgen', 'ldt', etc.)
            - 'archiname': Architecture name ('cktarchi', 'pace', etc.)
            - 'device': Torch device for the model
            - Other model-specific parameters
            
    Returns:
        Instantiated model moved to the specified device.
        
    Raises:
        ModuleNotFoundError: If specified model type or architecture doesn't exist.
        AttributeError: If required class is not found in module.
        
    Example:
        >>> args = {
        ...     'modeltype': 'cktgen',
        ...     'archiname': 'cktarchi',
        ...     'device': torch.device('cuda'),
        ...     'latent_dim': 64,
        ...     # ... other args
        ... }
        >>> model = get_model(args)
    """
    modeltype = args['modeltype']
    archiname = args['archiname']

    archi_module = importlib.import_module(f".architectures.{archiname}", package='models')
    Archi = archi_module.__getattribute__(f"{archiname.upper()}")

    if modeltype != 'cgan':
        model_module = importlib.import_module(f".modeltype.{modeltype}", package='models')
        Model = model_module.__getattribute__(f"{modeltype.upper()}")


    if archiname == 'cktarchi':
        encoder_module = importlib.import_module(f".architectures.ckt_encoder", package='models')
        Encoder = encoder_module.__getattribute__(f"Ckt_TransEncoder")
        decoder_module = importlib.import_module(f".architectures.ckt_decoder", package='models')
        Decoder = decoder_module.__getattribute__(f"Ckt_TransDecoder")
        encoder, decoder = Encoder(**args), Decoder(**args)
        archi = Archi(encoder, decoder, **args)
    else:
        archi = Archi(**args)

    if modeltype=='cktgen':  # and args['conditioned']
        spec_encoder_module = importlib.import_module(f".architectures.spec_encoder", package='models')
        Spec_Encoder = spec_encoder_module.__getattribute__(f"Spec_Encoder")
        spec_encoder = Spec_Encoder(**args)
        return Model(archi=archi, spec_encoder=spec_encoder, **args).to(args['device'])
    elif modeltype=='ldt':
        denoiser_module = importlib.import_module(f".modeltype.ldt_module.denoiser", package='models')
        denoiser_model = denoiser_module.__getattribute__(f"Denoiser")
        denoiser = denoiser_model(**args).to(args['device'])
        return Model(denoiser=denoiser, **args).to(args['device'])
    elif modeltype=='cvaegan':
        discriminator_module = importlib.import_module(f".modeltype.cvaegan", package='models')
        Discriminator = discriminator_module.__getattribute__(f"Discriminator")
        discriminator = Discriminator(**args)
        return Model(archi=archi, discriminator=discriminator, **args).to(args['device'])
    else:
        return Model(archi=archi, **args).to(args['device'])    
    
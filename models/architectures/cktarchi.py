"""CktArchi: Transformer-based encoder-decoder architecture for circuits.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module implements CktArchi, a transformer-based encoder-decoder architecture
optimized for circuit graph representation and generation. It provides the highest
quality results among all architectures but requires more computational resources.
"""


from torch import nn


class CKTARCHI(nn.Module):
    """Transformer-based encoder-decoder for circuit graphs.
    
    CktArchi combines a transformer encoder (Ckt_TransEncoder) with an
    autoregressive decoder (Ckt_TransDecoder) to learn and generate circuit
    topologies. It uses multi-head self-attention to capture long-range
    dependencies in circuit graphs.
    
    Attributes:
        encoder: Transformer encoder module (Ckt_TransEncoder).
        decoder: Transformer decoder module (Ckt_TransDecoder).
    """
    
    def __init__(self, encoder, decoder, **kwargs):
        """Initializes CktArchi with encoder and decoder modules.
        
        Args:
            encoder: Pre-instantiated encoder module.
            decoder: Pre-instantiated decoder module.
            **kwargs: Additional arguments (for compatibility).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, batch):
        """Encodes circuit graphs into latent representations.
        
        Args:
            batch: Dictionary containing circuit graph data.
            
        Returns:
            Dictionary with encoded latent representations and distributions.
        """
        return self.encoder(batch)

    def decode(self, args, batch):
        """Decodes latent representations into circuit graphs.
        
        Args:
            args: Configuration arguments for decoding.
            batch: Dictionary containing latent representations.
            
        Returns:
            Dictionary with decoded circuit outputs.
        """
        return self.decoder(args, batch)

    def compute_loss(self, batch):
        """Computes reconstruction loss from decoder.
        
        Args:
            batch: Dictionary containing ground truth and predictions.
            
        Returns:
            Dictionary with loss values.
        """
        return self.decoder.compute_loss(batch)



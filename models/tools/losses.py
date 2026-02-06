"""Loss functions for training (KL, InfoNCE, etc.).

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module implements custom loss functions used in CktGen:
- KLLoss: KL divergence between two Gaussian distributions
- InfoNCE_with_filtering: Contrastive loss with false negative filtering

References:
    KL divergence formulation:
    - https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    - https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
"""


import torch
import torch.nn.functional as F


class KLLoss:
    """KL divergence loss between two Gaussian distributions.
    
    Computes KL(q || p) where q and p are Gaussian distributions
    represented by their mean (mu) and log-variance (logvar).
    
    The KL divergence between two Gaussians N(mu_q, sigma_q²) and 
    N(mu_p, sigma_p²) is:
        KL = 0.5 * (log(sigma_p²/sigma_q²) + (sigma_q² + (mu_q - mu_p)²)/sigma_p² - 1)
    """
    
    def __call__(self, q, p):
        """Computes KL divergence KL(q || p).
        
        Args:
            q: Tuple of (mu_q, logvar_q) for distribution q.
            p: Tuple of (mu_p, logvar_p) for distribution p.
            
        Returns:
            Scalar KL divergence summed over all dimensions.
        """
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p  # log(sigma_q² / sigma_p²)
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.sum()

    def __repr__(self):
        return "KLLoss()"


class InfoNCE_with_filtering:
    """InfoNCE contrastive loss with false negative filtering.
    
    Implements the InfoNCE (normalized temperature-scaled cross entropy) loss
    for contrastive learning, with optional masking to filter false negatives
    (e.g., circuits with similar specifications that shouldn't be treated as
    negatives).
    
    Attributes:
        temperature: Temperature parameter for scaling similarities.
    """
    
    def __init__(self, temperature=0.7):
        """Initializes the InfoNCE loss.
        
        Args:
            temperature: Temperature parameter τ for scaling similarities.
                Lower values make the model more confident (sharper distribution).
        """
        self.temperature = temperature


    def get_sim_matrix(self, x, y):
        """Computes cosine similarity matrix between two sets of embeddings.
        
        Args:
            x: First set of embeddings [batch_size, embedding_dim].
            y: Second set of embeddings [batch_size, embedding_dim].
            
        Returns:
            Similarity matrix of shape [batch_size, batch_size] where
            sim[i, j] = cosine_similarity(x[i], y[j]).
        """
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix


    def __call__(self, x, y, mask=None):
        """Computes bidirectional InfoNCE loss with optional filtering.
        
        The loss encourages:
        - High similarity between matching pairs (x[i], y[i])
        - Low similarity between non-matching pairs (x[i], y[j]) for i ≠ j
        
        Args:
            x: First set of embeddings (e.g., circuit embeddings) 
                [batch_size, embedding_dim].
            y: Second set of embeddings (e.g., spec embeddings)
                [batch_size, embedding_dim].
            mask: Optional mask tensor [batch_size, batch_size] to filter 
                false negatives. Large negative values suppress pairs.
                
        Returns:
            Scalar contrastive loss averaged over both directions (x→y and y→x).
        """
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if mask is not None:
            # Filter false negatives (circuits with similar specs)
            # Large negative mask values → low probability after softmax
            sim_matrix = sim_matrix + mask

        # Labels: each x[i] should match with y[i]
        labels = torch.arange(bs, device=device)
        
        # Bidirectional loss: x→y and y→x
        total_loss = (F.cross_entropy(sim_matrix, labels, reduction='sum') + 
                      F.cross_entropy(sim_matrix.T, labels, reduction='sum')) / 2

        return total_loss


    def __repr__(self):
        return f"InfoNCE_with_filtering(temperature={self.temperature})"

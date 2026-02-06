"""Latent Diffusion Transformer (LDT) for high-quality circuit generation.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers import DDPMScheduler

class Spec_Embedder(nn.Module):
    """Specification embedder for LDT model.
    
    Encodes performance specifications into latent representations for
    conditioning the diffusion process.
    
    Args:
        num_gain_type: Number of discretized gain levels.
        num_pm_type: Number of discretized phase margin levels.
        num_bw_type: Number of discretized bandwidth levels.
        emb_dim: Embedding dimension.
        latent_dim: Output latent dimension.
    """
    def __init__(self, num_gain_type, num_pm_type, num_bw_type, emb_dim=128, latent_dim=64):
        super().__init__()

        self.gain_embed = nn.Embedding(num_gain_type, emb_dim)
        self.pm_embed   = nn.Embedding(num_pm_type, emb_dim)
        self.bw_embed   = nn.Embedding(num_bw_type, emb_dim)

        self.cond_proj = nn.Linear(emb_dim*3, emb_dim)
        self.fc_latent  = nn.Linear(emb_dim, latent_dim)
        
    def forward(self, batch):
        """Encode specifications to latent representations.
        
        Args:
            batch: Dictionary containing 'gains', 'bws', 'pms'.
            
        Returns:
            Spec latent embeddings of shape (batch_size, latent_dim).
        """
        gains, bws, pms = batch['gains'], batch['bws'], batch['pms']
        gain_embs = self.gain_embed(gains)
        bw_embs = self.bw_embed(bws)
        pm_embs = self.pm_embed(pms)

        conditions = torch.cat([gain_embs, bw_embs, pm_embs], dim=-1)
        conditions = self.cond_proj(conditions)

        spec_latents = self.fc_latent(conditions)
        return spec_latents

class LDT(nn.Module):
    """Latent Diffusion Transformer for circuit generation.
    
    High-quality circuit generation using diffusion models in latent space.
    Supports classifier-free guidance for better spec conditioning.
    
    Args:
        denoiser: Denoiser network module.
        device: Device to run the model on.
        num_train_timesteps: Number of diffusion timesteps during training.
        num_inference_timesteps: Number of timesteps during inference.
        latent_dim: Dimension of latent space.
        emb_dim: Embedding dimension for specs.
        guidance_scale: Classifier-free guidance scale (>1.0 enables guidance).
        beta_start: Starting value for noise schedule.
        beta_end: Ending value for noise schedule.
        beta_schedule: Type of noise schedule ('linear', 'scaled_linear', etc.).
        eta: DDIM eta parameter for stochasticity.
        guidance_uncod_prob: Probability of unconditional training.
        num_gain_type: Number of gain levels.
        num_pm_type: Number of phase margin levels.
        num_bw_type: Number of bandwidth levels.
        **kwargs: Additional arguments.
    """
    def __init__(
        self,
        denoiser,
        device,
        num_train_timesteps=1000,
        num_inference_timesteps=50,
        latent_dim=64,
        emb_dim=128,
        guidance_scale=7.5,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        eta=0.0,
        guidance_uncod_prob=0.1,
        num_gain_type=10,  
        num_pm_type=10,
        num_bw_type=10,
        **kwargs
    ):
        super().__init__()
        self.vae = None
        self.device = device
        self.denoiser = denoiser
        self.latent_dim = [1, latent_dim]
        self.spec_embedder = Spec_Embedder(
            num_gain_type=num_gain_type,  
            num_pm_type=num_pm_type,    
            num_bw_type=num_bw_type,    
            emb_dim=emb_dim,
            latent_dim=self.latent_dim[-1]
        )
        self.num_train_timesteps = num_train_timesteps

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type='fixed_small',
            clip_sample=False,
            prediction_type='sample',
        )
        
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        
        self.num_inference_timesteps = num_inference_timesteps
        self.eta = eta
        self.guidance_uncod_prob = guidance_uncod_prob
        
    def load_vae(self, vae_pth):
        """Load pretrained VAE encoder and freeze its parameters.
        
        Args:
            vae_pth: Path to saved VAE checkpoint.
        """
        vae_ckpt = torch.load(vae_pth, map_location='cpu')
        self.vae = vae_ckpt['model'].to(self.device)
        
        # fraze vae parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        # vae = torch.load(args['vae_pth'], map_location='cpu').to(args['device'])

    def compute_loss(self, batch):
        """Compute diffusion training loss.
        
        Args:
            batch: Training batch containing circuit latents and specifications.
            
        Returns:
            MSE loss between ground truth and denoised latents.
        """
        ckt_latents = batch['ckt_latents'].permute(1, 0, 2)             # (bsz, 1, latent_dim)
        spec_latents = self.spec_embedder(batch).unsqueeze(1)           # (bsz, 1, latent_dim)

        noise = torch.randn_like(ckt_latents)       # (bsz, 1, latent_dim)
        bsz = ckt_latents.shape[0]
        timesteps = torch.randint(
            low=0, 
            high=self.num_train_timesteps, 
            size=(bsz,), 
            device=ckt_latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(ckt_latents.clone(), noise, timesteps) # (bsz, 1, latent_dim)

        # print('noisy latent size: ', noisy_latents.size())
        for i in range(bsz):
            if np.random.rand(1) < self.guidance_uncod_prob:
                spec_latents[i] = torch.zeros_like(spec_latents[i])
        
        denoise_latents = self.denoiser( # (bsz, 1, latent_dim)
            sample=noisy_latents,
            timestep=timesteps, 
            condition=spec_latents
        )
        # print('denoise pred size: ', denoise_latents.size())
        loss = F.mse_loss(ckt_latents, denoise_latents)
        return loss

    def forward(self, args, batch):
        """Generate circuit latents using diffusion sampling.
        
        Args:
            args: Additional arguments (unused but kept for compatibility).
            batch: Batch containing performance specifications.
            
        Returns:
            Generated circuit latents of shape (batch_size, 1, latent_dim).
        """
        spec_latents = self.spec_embedder(batch).unsqueeze(1) # (bsz, 1, latent_dim)
        bsz = spec_latents.shape[0]

        if self.do_classifier_free_guidance:
            condition = torch.cat((torch.zeros_like(spec_latents), spec_latents))
        else:
            condition = spec_latents

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=condition.device,
            dtype=torch.float
        ) # (bsz, 1, latent_dim)

        latents = latents * self.scheduler.init_noise_sigma         # (bsz, 1, latent_dim)
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(condition.device)

        extra_step_kwargs = {}
        if 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs['eta'] = self.eta

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            ) # (2 * bsz, 1, latent_dim)

            pred = self.denoiser(
                sample=latent_model_input,
                timestep=t, 
                condition=condition
            ) # (2 * bsz, 1, latent_dim)
            
            if self.do_classifier_free_guidance:
                uncond_pred, cond_pred = pred.chunk(2)
                pred = uncond_pred + self.guidance_scale * (cond_pred - uncond_pred)

            latents = self.scheduler.step(pred, t, latents, **extra_step_kwargs).prev_sample
        # print('denoise pred size: ', latents.size())
        return latents # (bsz, 1, latent_dim)

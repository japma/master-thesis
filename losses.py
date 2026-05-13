"""Loss utilities."""

import torch
import torch.nn
from torch import nn


def vae_loss(images, recon, mu, logvar, beta=1.0):
    """ELBO loss: reconstruction (MSE) + β · KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, images, reduction="sum") / images.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def compute_nll(model, z_target, z_cond):
    """Evaluate CSPN log-likelihood and return (nll_loss, mean_log_prob)."""
    log_prob = model(z_cond, z_target)
    loss = -log_prob.mean()
    return loss, log_prob.mean().item()

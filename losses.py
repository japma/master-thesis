"""Loss utilities."""

import torch
import torch.nn
from torch import nn


def vae_loss(
    images: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_fn: nn.Module = nn.MSELoss(),
):
    """ELBO loss: reconstruction (MSE) + β · KL divergence."""
    recon_loss = recon_loss_fn(recon, images, reduction="sum") / images.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def negative_log_likelihood_loss(
    outputs: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for SPN outputs."""
    return -outputs.mean()

"""Loss utilities."""

import torch
import torch.nn
from torch import nn


class HybridLoss(nn.Module):
    """Combines MSE and L1 loss: 0.5*MSE + 0.5*L1. Better detail preservation than MSE alone."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, recon, target):
        return 0.5 * self.mse(recon, target) + 0.5 * self.l1(recon, target)


def vae_loss(
    images: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_fn: nn.Module = nn.MSELoss(),
):
    """ELBO loss: reconstruction + β · KL divergence.

    KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    where logvar = log(sigma^2)
    """
    recon_loss = recon_loss_fn(recon, images)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def negative_log_likelihood_loss(
    outputs: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for SPN outputs."""
    return -outputs.mean()

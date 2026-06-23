"""Loss utilities."""

import torch
import torch.nn
from torch import nn


class HybridLoss(nn.Module):
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
    recon_loss = recon_loss_fn(recon, images)
    return recon_loss, recon_loss, torch.tensor(0.0)


def beta_vae_loss(
    images: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_fn: nn.Module = nn.MSELoss(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = recon_loss_fn(recon, images)

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def negative_log_likelihood_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for SPN outputs."""
    # correct_class_ll = outputs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # return -correct_class_ll.mean()
    return -outputs.mean()


def get_ae_loss_fn(loss_type: str) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "hybrid":
        return HybridLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)

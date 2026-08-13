from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    total: torch.Tensor


def kl_loss_fn(
    mu: torch.Tensor, log_var: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    log_var = log_var.clamp(-30.0, 20.0)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl_per_sample = kl_per_dim.sum(dim=-1)
    return kl_per_sample.mean()


def kl_per_dimension(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(z_i|x) || p(z_i)) per latent dimension, averaged over the batch."""
    log_var = log_var.clamp(-30.0, 20.0)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean(dim=0)

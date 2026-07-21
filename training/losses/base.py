from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    total: torch.Tensor


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(-30.0, 20.0)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean()

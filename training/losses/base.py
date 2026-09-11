import math
from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    total: torch.Tensor


def kl_loss_fn(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    free_bits: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL to a standard normal prior. `mask` is a boolean over latent dimensions; the
    ones it drops are left to some other prior (see `anchor_kl_per_dim`) rather than
    pulled towards zero by this term as well."""
    log_var = log_var.clamp(-30.0, 20.0)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    if mask is not None:
        kl_per_dim = kl_per_dim[:, mask]
    kl_per_sample = kl_per_dim.sum(dim=-1)
    return kl_per_sample.mean()


def anchor_kl_per_dim(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    anchor: torch.Tensor,
    prior_std: float,
) -> torch.Tensor:
    """Per-dimension KL(q(z|x) || N(anchor, prior_std^2)), shape `(batch, dims)`.

    A full KL rather than a squared distance to the anchor on purpose: it pins the
    posterior *variance* to `prior_std^2` as well as its mean, which is what stops the
    dimension collapsing to a point mass that a downstream circuit's Gaussian leaf
    cannot represent.
    """
    log_var = log_var.clamp(-30.0, 20.0)
    prior_log_var = 2 * math.log(prior_std)
    return 0.5 * (
        prior_log_var
        - log_var
        + (log_var.exp() + (mu - anchor).pow(2)) / (prior_std**2)
        - 1.0
    )


def kl_per_dimension(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(z_i|x) || p(z_i)) per latent dimension, averaged over the batch."""
    log_var = log_var.clamp(-30.0, 20.0)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean(dim=0)

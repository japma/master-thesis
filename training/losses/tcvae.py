from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from training.losses.base import LossOutput


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)


def log_density_gaussian(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Elementwise log density of a diagonal Gaussian N(mu, exp(logvar))."""
    std = torch.exp(0.5 * logvar)
    return torch.distributions.Normal(mu, std).log_prob(z)


@dataclass
class TCVAELossOutput(LossOutput):
    recon: torch.Tensor
    kl: torch.Tensor
    mi: torch.Tensor
    tc: torch.Tensor
    dwkl: torch.Tensor


class BetaTCVAELoss(nn.Module):
    """Beta-TCVAE loss

    Args:
        dataset_size (int): Number of examples in the dataset.
        alpha (float):
        beta (float):
        gamma (float):
        free_bits (float):
    """

    def __init__(
        self,
        dataset_size: int,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        free_bits: float = 0.5,
    ) -> None:
        super().__init__()
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.free_bits = free_bits

    def _decompose_kl(
        self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z.shape[0]

        log_qz_given_x = log_density_gaussian(z, mu, logvar).sum(dim=1)

        mu_expanded = mu.unsqueeze(0)
        logvar_expanded = logvar.unsqueeze(0)
        z_expanded = z.unsqueeze(1)
        log_qz_matrix = log_density_gaussian(z_expanded, mu_expanded, logvar_expanded)

        log_norm = torch.log(
            torch.tensor(
                batch_size * self.dataset_size, dtype=torch.float32, device=z.device
            )
        )

        log_qz = torch.logsumexp(log_qz_matrix.sum(dim=2), dim=1) - log_norm
        log_qz_product = (torch.logsumexp(log_qz_matrix, dim=1) - log_norm).sum(dim=1)

        log_pz = log_density_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(
            dim=1
        )

        # TODO maybe those means here are the problem
        mutual_info = (log_qz_given_x - log_qz).mean()
        total_correlation = (log_qz - log_qz_product).mean()
        dimension_wise_kl = (log_qz_product - log_pz).clamp(min=self.free_bits).mean()

        return mutual_info, total_correlation, dimension_wise_kl

    def forward(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        beta: float | None = None,
    ) -> TCVAELossOutput:
        recon_loss = F.binary_cross_entropy_with_logits(recon, images, reduction="mean")

        logvar = logvar.clamp(-30.0, 20.0)
        mi, tc, dwkl = self._decompose_kl(z, mu, logvar)

        effective_beta = beta if beta is not None else self.beta
        kl_loss = self.alpha * mi + effective_beta * tc + self.gamma * dwkl

        return TCVAELossOutput(
            total=recon_loss + kl_loss,
            recon=recon_loss,
            kl=kl_loss,
            mi=mi,
            tc=tc,
            dwkl=dwkl,
        )

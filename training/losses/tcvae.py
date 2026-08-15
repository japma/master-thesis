import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.base import LossOutput
from training.losses.perceptual import VGGPerceptualLoss

LOG_2PI = math.log(2.0 * math.pi)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)


def log_density_gaussian(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Elementwise log density of a diagonal Gaussian N(mu, exp(logvar))."""
    return -0.5 * (LOG_2PI + logvar + (z - mu).pow(2) / torch.exp(logvar))


@dataclass
class TCVAELossOutput(LossOutput):
    recon: torch.Tensor
    kl: torch.Tensor
    mi: torch.Tensor
    tc: torch.Tensor
    dwkl: torch.Tensor
    perceptual: torch.Tensor


class BetaTCVAELoss(nn.Module):
    """Beta-TCVAE loss

    Args:
        dataset_size (int): Number of examples in the dataset.
        alpha (float):
        beta (float):
        gamma (float):
        free_bits (float):
        lambda_perceptual (float): weight of the VGG perceptual term; 0 disables it.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        free_bits: float = 0.5,
        lambda_perceptual: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.free_bits = free_bits
        self.lambda_perceptual = lambda_perceptual
        self.perceptual = VGGPerceptualLoss() if lambda_perceptual > 0 else None

    def _decompose_kl(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        dataset_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z.shape[0]

        log_qz_given_x = log_density_gaussian(z, mu, log_var).sum(dim=1)

        mu_expanded = mu.unsqueeze(0)
        log_var_expanded = log_var.unsqueeze(0)
        z_expanded = z.unsqueeze(1)

        log_qz_matrix = log_density_gaussian(z_expanded, mu_expanded, log_var_expanded)

        log_norm = torch.log(
            torch.tensor(
                batch_size * dataset_size, dtype=torch.float32, device=z.device
            )
        )

        log_qz_joint = log_qz_matrix.sum(dim=2)
        log_qz = torch.logsumexp(log_qz_joint, dim=1) - log_norm

        log_qz_marginals = torch.logsumexp(log_qz_matrix, dim=1) - log_norm
        log_qz_product = log_qz_marginals.sum(dim=1)

        log_pz_per_dim = log_density_gaussian(
            z,
            torch.zeros_like(z),
            torch.zeros_like(z),
        )
        log_pz = log_pz_per_dim.sum(dim=1)

        mutual_info = (log_qz_given_x - log_qz).mean()
        total_correlation = (log_qz - log_qz_product).mean()
        dwkl_per_dim = log_qz_marginals - log_pz_per_dim

        if self.free_bits > 0:
            dwkl_per_dim = dwkl_per_dim.clamp(min=self.free_bits)

        dimension_wise_kl = dwkl_per_dim.sum(dim=1).mean()

        return mutual_info, total_correlation, dimension_wise_kl

    def forward(
        self,
        target: torch.Tensor,
        model_output: VAEForwardOutput,
        dataset_size: int,
        beta: float | None = None,
    ) -> TCVAELossOutput:
        recon_loss = (
            F.binary_cross_entropy_with_logits(
                model_output.reconstructed,
                target,
                reduction="none",
            )
            .flatten(1)
            .sum(dim=1)
            .mean()
        )

        log_var = model_output.log_var.clamp(-30.0, 20.0)
        mi, tc, dwkl = self._decompose_kl(
            model_output.latent, model_output.mu, log_var, dataset_size=dataset_size
        )

        effective_beta = beta if beta is not None else self.beta
        kl_loss = self.alpha * mi + effective_beta * tc + self.gamma * dwkl

        recon_img = torch.sigmoid(model_output.reconstructed)
        if self.perceptual is not None:
            perc = self.perceptual(recon=recon_img, target=target)
        else:
            perc = torch.tensor(0.0, device=target.device)

        total_loss = recon_loss + kl_loss + self.lambda_perceptual * perc

        return TCVAELossOutput(
            total=total_loss,
            recon=recon_loss,
            kl=kl_loss,
            mi=mi,
            tc=tc,
            dwkl=dwkl,
            perceptual=perc,
        )

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.base import LossOutput, kl_loss_fn
from training.losses.perceptual import VGGPerceptualLoss


@dataclass
class VAELossOutput(LossOutput):
    recon: torch.Tensor
    kl: torch.Tensor
    perceptual: torch.Tensor


class VAELoss(nn.Module):
    """VAE training loss: recon + KL + perceptual."""

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.5,
        lambda_perceptual: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.lambda_perceptual = lambda_perceptual
        self.perceptual = VGGPerceptualLoss() if lambda_perceptual > 0 else None

    def forward(
        self,
        images: torch.Tensor,
        model_outputs: VAEForwardOutput,
        beta: float | None = None,
    ) -> VAELossOutput:

        recon_loss = F.binary_cross_entropy_with_logits(
            model_outputs.reconstructed, images, reduction="mean"
        )

        kl_loss = kl_loss_fn(
            model_outputs.mu, model_outputs.log_var, free_bits=self.free_bits
        )

        recon_img = torch.sigmoid(model_outputs.reconstructed)
        if self.perceptual is not None:
            perc = self.perceptual(recon=recon_img, target=images)
        else:
            perc = torch.tensor(0.0, device=images.device)

        effective_beta = beta if beta is not None else self.beta
        total_loss = (
            recon_loss + effective_beta * kl_loss + self.lambda_perceptual * perc
        )
        return VAELossOutput(
            total=total_loss,
            recon=recon_loss,
            kl=kl_loss,
            perceptual=perc,
        )

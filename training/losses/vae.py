from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.base import LossOutput, kl_loss
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
        # TODO use free bits
        self.free_bits = free_bits
        self.lambda_perceptual = lambda_perceptual
        self.perceptual = VGGPerceptualLoss() if lambda_perceptual > 0 else None

    def forward(
        self,
        images: torch.Tensor,
        model_outputs: VAEForwardOutput,
        beta: float | None = None,
    ) -> VAELossOutput:

        recon = F.binary_cross_entropy_with_logits(
            model_outputs.reconstructed, images, reduction="mean"
        )

        kl = kl_loss(model_outputs.mu, model_outputs.log_var)

        recon_img = torch.sigmoid(model_outputs.reconstructed)
        if self.perceptual is not None:
            perc = self.perceptual(images, recon_img)
        else:
            perc = torch.tensor(0.0, device=images.device)

        effective_beta = beta if beta is not None else self.beta
        total = recon + effective_beta * kl + self.lambda_perceptual * perc
        return VAELossOutput(
            total=total,
            recon=recon,
            kl=kl,
            perceptual=perc,
        )

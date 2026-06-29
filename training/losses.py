"""losses.py

Provides:
  - BetaVAELoss        — β-VAE loss (recon + KL)
  - VGGPerceptualLoss  — frozen VGG16 feature-space L1 loss
  - VAELoss            — combined recon + KL + perceptual loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class BetaVAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, free_bits: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits

    def forward(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.binary_cross_entropy_with_logits(recon, images, reduction="mean")

        logvar = logvar.clamp(-30.0, 20.0)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.clamp(min=self.free_bits).mean()

        effective_beta = beta if beta is not None else self.beta
        return recon_loss + effective_beta * kl_loss, recon_loss, kl_loss


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss using a frozen VGG16."""

    _FEATURE_LAYERS: tuple[int, ...] = (9, 16)  # relu2_2, relu3_3 in VGG16 features

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg.children())[: self._FEATURE_LAYERS[-1] + 1])
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # type: ignore[operator]

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._normalise(x)
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self._FEATURE_LAYERS:
                feats.append(x)
        return feats

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Both recon and target must be in [0, 1]."""
        recon_feats = self._features(recon)
        with torch.no_grad():
            target_feats = self._features(target)
        return sum(F.l1_loss(r, t) for r, t in zip(recon_feats, target_feats)) / len(
            recon_feats
        )  # type: ignore[return-value]


class VAELossOutput(dict):
    total: torch.Tensor
    recon: torch.Tensor  # BCE reconstruction loss
    kl: torch.Tensor  # KL divergence
    perceptual: torch.Tensor  # VGG perceptual loss


class VAELoss(nn.Module):
    """VAE training loss: recon + KL + perceptual."""

    def __init__(
        self,
        beta_vae_loss: BetaVAELoss,
        lambda_perceptual: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta_vae = beta_vae_loss
        self.perceptual = VGGPerceptualLoss()
        self.lambda_perceptual = lambda_perceptual

    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float | None = None,
    ) -> VAELossOutput:
        total_beta_vae, recon_loss, kl_loss = self.beta_vae(
            images, logits, mu, logvar, beta
        )

        recon_img = torch.sigmoid(logits)
        p_loss = self.perceptual(recon_img, images)

        total = total_beta_vae + self.lambda_perceptual * p_loss

        return VAELossOutput(
            total=total,
            recon=recon_loss,
            kl=kl_loss,
            perceptual=p_loss,
        )


def negative_log_likelihood_loss(
    outputs: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for SPN outputs."""
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

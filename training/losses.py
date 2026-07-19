"""losses.py

Provides:
  - BetaVAELoss        — β-VAE loss (recon + KL)
  - VGGPerceptualLoss  — frozen VGG16 feature-space L1 loss
  - VAELoss            — combined recon + KL + perceptual loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class BetaVAEComponents(TypedDict):
    recon: torch.Tensor
    kl: torch.Tensor


class VAEComponents(TypedDict):
    recon: torch.Tensor
    kl: torch.Tensor
    perceptual: torch.Tensor


class TCVAEComponents(TypedDict):
    recon: torch.Tensor
    kl: torch.Tensor
    mi: torch.Tensor
    tc: torch.Tensor
    dwkl: torch.Tensor


TComponents = TypeVar("TComponents", bound=dict)


@dataclass
class LossOutput[TComponents: dict]:
    total: torch.Tensor
    components: TComponents

    def __getitem__(self, key: str) -> torch.Tensor:
        if key == "total":
            return self.total
        return self.components[key]

    def to_metrics(self) -> dict[str, float]:
        metrics = {"total": self.total.item()}
        metrics.update({key: value.item() for key, value in self.components.items()})
        return metrics


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
    ) -> LossOutput:
        recon_loss = F.binary_cross_entropy_with_logits(recon, images, reduction="mean")

        logvar = logvar.clamp(-30.0, 20.0)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.clamp(min=self.free_bits).mean()

        effective_beta = beta if beta is not None else self.beta
        total = recon_loss + effective_beta * kl_loss
        return LossOutput(total=total, components={"recon": recon_loss, "kl": kl_loss})


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss using a frozen VGG16."""

    _FEATURE_LAYERS: tuple[int, int] = (9, 16)  # relu2_2, relu3_3 in VGG16 features

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
        # pyrefly: ignore [bad-return]
        return sum(
            F.l1_loss(r, t) for r, t in zip(recon_feats, target_feats, strict=False)
        ) / len(recon_feats)


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
    ) -> LossOutput:
        beta_vae_out = self.beta_vae(images, logits, mu, logvar, beta)

        recon_img = torch.sigmoid(logits)
        p_loss = self.perceptual(recon_img, images)

        total = beta_vae_out.total + self.lambda_perceptual * p_loss

        return LossOutput(
            total=total,
            components=VAEComponents(
                recon=beta_vae_out.components["recon"],
                kl=beta_vae_out.components["kl"],
                perceptual=p_loss,
            ),
        )


def negative_log_likelihood_loss(outputs: torch.Tensor) -> LossOutput:
    """Negative log-likelihood loss for SPN outputs."""
    nll = -outputs.mean()
    return LossOutput(total=nll)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)


def log_density_gaussian(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Elementwise log density of a diagonal Gaussian N(mu, exp(logvar))."""
    std = torch.exp(0.5 * logvar)
    return torch.distributions.Normal(mu, std).log_prob(z)


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
    ) -> LossOutput:
        recon_loss = F.binary_cross_entropy_with_logits(recon, images, reduction="mean")

        logvar = logvar.clamp(-30.0, 20.0)
        mi, tc, dwkl = self._decompose_kl(z, mu, logvar)

        effective_beta = beta if beta is not None else self.beta
        kl_loss = self.alpha * mi + effective_beta * tc + self.gamma * dwkl

        return LossOutput(
            total=recon_loss + kl_loss,
            components=TCVAEComponents(
                recon=recon_loss,
                kl=kl_loss,
                mi=mi,
                tc=tc,
                dwkl=dwkl,
            ),
        )

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoder.supervised_vae import SupervisedVAEForwardOutput
from training.losses.vae import VAELoss, VAELossOutput


@dataclass
class SupervisedVAELossOutput(VAELossOutput):
    """`classification` is the summed cross-entropy the latent blocks are trained on;
    `per_factor` keeps the individual terms so wandb shows which attribute is hard."""

    classification: torch.Tensor
    per_factor: list[torch.Tensor]


class SupervisedVAELoss(nn.Module):
    """VAE loss plus a cross-entropy per label factor, each read off its own block of
    the latent. At gamma 0 this is exactly the wrapped `VAELoss`."""

    def __init__(self, vae_loss: VAELoss, gamma: float = 1.0) -> None:
        super().__init__()
        self.vae_loss = vae_loss
        self.gamma = gamma

    def forward(
        self,
        images: torch.Tensor,
        model_outputs: SupervisedVAEForwardOutput,
        labels: torch.Tensor,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> SupervisedVAELossOutput:
        vae = self.vae_loss(images, model_outputs, beta=beta)

        targets = labels if labels.ndim > 1 else labels.unsqueeze(1)
        if targets.shape[1] != len(model_outputs.logits):
            raise ValueError(
                f"model has {len(model_outputs.logits)} label factors but the batch "
                f"carries {targets.shape[1]} targets per sample"
            )

        per_factor = [
            F.cross_entropy(logits, targets[:, i].long())
            for i, logits in enumerate(model_outputs.logits)
        ]
        classification = torch.stack(per_factor).sum()

        effective_gamma = gamma if gamma is not None else self.gamma
        return SupervisedVAELossOutput(
            total=vae.total + effective_gamma * classification,
            recon=vae.recon,
            kl=vae.kl,
            perceptual=vae.perceptual,
            classification=classification,
            per_factor=per_factor,
        )

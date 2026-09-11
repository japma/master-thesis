from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoder.anchored_vae import AnchoredVAEForwardOutput
from training.losses.anchors import anchor_tables, anchor_targets
from training.losses.base import anchor_kl_per_dim
from training.losses.vae import VAELoss, VAELossOutput
from utils.config import SupervisionConfig


@dataclass
class AnchoredVAELossOutput(VAELossOutput):
    anchor: torch.Tensor
    classification: torch.Tensor
    per_anchor: list[torch.Tensor]
    per_head: list[torch.Tensor]


class AnchoredVAELoss(nn.Module):
    anchor_dims: torch.Tensor
    free_mask: torch.Tensor

    def __init__(
        self,
        vae_loss: VAELoss,
        supervision: SupervisionConfig,
        latent_dim: int,
        gamma: float = 1.0,
        anchor_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.vae_loss = vae_loss
        self.gamma = gamma
        self.anchor_weight = anchor_weight
        self.supervision = supervision
        self.anchor_std = supervision.anchor_std
        self.anchored_factors = supervision.anchored_factors
        self.head_factors = supervision.head_factors
        self.anchor_block_dims = [supervision.dims[i] for i in self.anchored_factors]

        anchored = supervision.anchored_dims()
        free_mask = torch.ones(latent_dim, dtype=torch.bool)
        free_mask[anchored] = False
        self.register_buffer("anchor_dims", torch.tensor(anchored, dtype=torch.long))
        self.register_buffer("free_mask", free_mask)

        for i, table in zip(
            self.anchored_factors, anchor_tables(supervision), strict=True
        ):
            self.register_buffer(f"anchor_table_{i}", table)

    def tables(self) -> list[torch.Tensor]:
        return [getattr(self, f"anchor_table_{i}") for i in self.anchored_factors]

    @torch.no_grad()
    def anchor_rmse(self, mu: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
        targets = labels if labels.ndim > 1 else labels.unsqueeze(1)
        anchor = anchor_targets(self.tables(), self.anchored_factors, targets.long())
        error = (mu[:, self.anchor_dims] - anchor).pow(2)
        return [
            block.mean().sqrt() for block in error.split(self.anchor_block_dims, dim=1)
        ]

    def forward(
        self,
        images: torch.Tensor,
        model_outputs: AnchoredVAEForwardOutput,
        labels: torch.Tensor,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> AnchoredVAELossOutput:
        vae = self.vae_loss(images, model_outputs, beta=beta, kl_mask=self.free_mask)

        targets = labels if labels.ndim > 1 else labels.unsqueeze(1)
        num_factors = len(self.supervision.dims)
        if targets.shape[1] != num_factors:
            raise ValueError(
                f"model has {num_factors} label factors but the batch carries "
                f"{targets.shape[1]} targets per sample"
            )

        anchor = anchor_targets(self.tables(), self.anchored_factors, targets.long())
        per_dim = anchor_kl_per_dim(
            model_outputs.mu[:, self.anchor_dims],
            model_outputs.log_var[:, self.anchor_dims],
            anchor,
            self.anchor_std,
        )
        per_anchor = [
            block.sum(dim=1).mean()
            for block in per_dim.split(self.anchor_block_dims, dim=1)
        ]
        anchor_kl = torch.stack(per_anchor).sum()

        per_head = [
            F.cross_entropy(logits, targets[:, i].long())
            for logits, i in zip(model_outputs.logits, self.head_factors, strict=True)
        ]
        classification = (
            torch.stack(per_head).sum()
            if per_head
            else torch.zeros((), device=images.device)
        )

        effective_gamma = gamma if gamma is not None else self.gamma
        return AnchoredVAELossOutput(
            total=vae.total
            + effective_gamma * (self.anchor_weight * anchor_kl + classification),
            recon=vae.recon,
            kl=vae.kl,
            perceptual=vae.perceptual,
            anchor=anchor_kl,
            classification=classification,
            per_anchor=per_anchor,
            per_head=per_head,
        )

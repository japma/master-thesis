from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkForSPN(nn.Module):
    """Neural parameter function for SPFlow conditional Normal leaves.

    Simplified single-layer version for debugging and performance.
    """

    def __init__(
        self,
        latent_size: int,
        num_labels: int,
        num_components: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.num_components = num_components

        self.backbone = nn.Linear(num_labels, hidden_dim)
        self.loc_head = nn.Linear(hidden_dim, latent_size * num_components)
        self.scale_head = nn.Linear(hidden_dim, latent_size * num_components)

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence.dim() == 1:
            evidence = evidence.unsqueeze(1)

        labels = evidence[:, 0].long().clamp(min=0, max=self.num_labels - 1)
        h = F.silu(self.backbone(F.one_hot(labels, self.num_labels).float()))

        loc = self.loc_head(h).view(-1, self.latent_size, self.num_components, 1)
        raw_scale = self.scale_head(h).view(
            -1, self.latent_size, self.num_components, 1
        )
        scale = F.softplus(raw_scale) + 1e-3

        return {"loc": loc, "scale": scale}

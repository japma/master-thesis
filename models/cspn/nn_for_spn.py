from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkForSPN(nn.Module):
    """Neural parameter function for SPFlow conditional Normal leaves."""

    def __init__(
        self,
        latent_size: int,
        num_labels: int,
        num_components: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.num_components = num_components

        self.label_embedding = nn.Embedding(num_labels, embedding_dim)

        layers = []
        in_dim = embedding_dim
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.loc_head = nn.Linear(hidden_dim, latent_size * num_components)
        self.scale_head = nn.Linear(hidden_dim, latent_size * num_components)

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence.dim() == 1:
            evidence = evidence.unsqueeze(1)

        labels = evidence[:, 0].long().clamp(min=0, max=self.num_labels - 1)
        h = self.backbone(self.label_embedding(labels))

        loc = self.loc_head(h).view(-1, self.latent_size, self.num_components, 1)
        raw_scale = self.scale_head(h).view(
            -1, self.latent_size, self.num_components, 1
        )
        scale = F.softplus(raw_scale) + 1e-3

        return {"loc": loc, "scale": scale}

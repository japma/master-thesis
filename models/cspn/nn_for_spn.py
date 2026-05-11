from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class NeuralNetworkForSPN(nn.Module):
    """Neural parameter function for SPFlow conditional Normal leaves."""

    def __init__(
        self,
        conditional_dim: int,
        latent_dim: int,
        num_leaves: int,
        num_layers: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.conditional_dim = conditional_dim
        self.latent_dim = latent_dim
        self.num_leaves = num_leaves

        layers = []
        in_dim = conditional_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        out_dim = latent_dim * num_leaves
        self.loc_head = nn.Linear(hidden_dim, out_dim)
        self.scale_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        h = self.backbone(z)
        shape = (B, self.latent_dim, self.num_leaves)

        loc = self.loc_head(h).view(shape)
        scale = self.scale_head(h).view(shape).exp().clamp(min=1e-4)
        return loc, scale

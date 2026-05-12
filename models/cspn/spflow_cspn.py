"""Conditional Sum-Product Network using SPFlow's Einet."""

from __future__ import annotations

import torch
import torch.nn as nn

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.zoo.einet import Einet

from .nn_for_spn import NeuralNetworkForSPN


class SPFlowCSPN(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_leaves = 10
        self.num_reps = 5

        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.nn = NeuralNetworkForSPN(
            conditional_dim=latent_dim,
            latent_dim=latent_dim,
            num_leaves=self.num_leaves,
            num_layers=3,
            hidden_dim=256,
        )
        leaves = [
            Normal(
                scope=Scope([i]),
            )
            for i in range(latent_dim)
        ]

        # TODO increase depth to max
        self.einet = Einet(
            leaf_modules=list(leaves),
            num_classes=num_classes,
            num_leaves=self.num_leaves,
        )

    def forward(self, labels: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        loc, scale = self.nn(label_emb)

        log_prob = self.einet.log_likelihood(z)
        return log_prob

    def sample_all_classes(
        self,
        num_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return latent samples for all classes.

        Shape: (num_classes, num_samples, latent_dim)
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        device = device or self.label_embedding.weight.device
        class_ids = torch.arange(self.num_classes, device=device)
        class_cond = self.label_embedding(class_ids)
        class_loc, class_scale = self.nn(class_cond)

        mean = class_loc.mean(dim=-1)
        std = class_scale.mean(dim=-1)

        noise = torch.randn(
            self.num_classes, num_samples, self.latent_dim, device=device
        )
        return mean.unsqueeze(1) + noise * std.unsqueeze(1)

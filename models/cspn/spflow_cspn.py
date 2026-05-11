"""Conditional Sum-Product Network using SPFlow's Einet."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.distributions import Normal as TorchNormal

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.zoo.einet import Einet

from .nn_for_spn import NeuralNetworkForSPN

logger = logging.getLogger(__name__)


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

        self.einet = Einet(
            leaf_modules=list(leaves),
            num_classes=num_classes,
            num_leaves=self.num_leaves,
        )

        logger.info(
            "SPFlowCSPN: latent_dim=%d, num_classes=%d", latent_dim, num_classes
        )

    def forward(self, labels: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        loc, scale = self.nn(label_emb)

        log_prob = self.einet.log_likelihood(z)
        return log_prob

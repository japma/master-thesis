"""Conditional Sum-Product Network using SPFlow's Einet."""

from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.zoo.einet import Einet

from . import AbstractCSPN
from .nn_for_spflow import NeuralNetworkForSPFlow


class SPFlowCSPN(AbstractCSPN, ABC):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self._num_classes = num_classes
        self.num_leaves = 10
        self.num_reps = 5

        self.nn = NeuralNetworkForSPFlow(
            conditional_dim=num_classes,
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

    @property
    def num_classes(self) -> int:
        """Number of classes for one-hot encoding."""
        return self._num_classes
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Convert labels to one-hot encoding internally
        context = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        loc, scale = self.nn(context)

        log_prob = self.einet.log_likelihood(z)
        return log_prob

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

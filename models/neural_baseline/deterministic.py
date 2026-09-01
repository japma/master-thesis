"""Deterministic neural baseline."""

import math

import torch
from torch import nn

from models.cspn.psinet.label_encoder import build_label_encoder
from models.neural_baseline.abstract_nn import AbstractNeuralBaseline
from utils.config import NeuralBaselineConfig


class DeterministicBaseline(AbstractNeuralBaseline):
    def __init__(self, config: NeuralBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.num_vars = config.num_vars
        self.encoder = build_label_encoder(config.encoder_config)

        layers: list[nn.Module] = []
        in_dim = self.encoder.num_classes
        for h_dim in config.h_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, config.num_vars)

        self.register_buffer(
            "log_std", torch.tensor(math.log(config.fixed_std), dtype=torch.float32)
        )

    def _mean(self, labels: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.trunk(self.encoder(labels)))

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """log p(z | labels), shape [B]."""
        mean = self._mean(labels)
        normalized = (z - mean) / self.log_std.exp()
        return (-0.5 * (normalized**2 + math.log(2 * math.pi)) - self.log_std).sum(-1)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Predicted latent per label row, shape [B, num_vars]."""
        return self._mean(labels)

    def get_config(self) -> dict:
        return self.config.model_dump(mode="json")

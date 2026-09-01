"""Mixture density neural baseline."""

import math

import torch
from torch import nn

from models.cspn.psinet.label_encoder import build_label_encoder
from models.neural_baseline.abstract_nn import AbstractNeuralBaseline
from utils.config import NeuralBaselineConfig


class MixtureDensityBaseline(AbstractNeuralBaseline):
    def __init__(self, config: NeuralBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.num_vars = config.num_vars
        self.num_components = config.num_components
        self.encoder = build_label_encoder(config.encoder_config)

        layers: list[nn.Module] = []
        in_dim = self.encoder.num_classes
        for h_dim in config.h_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        out_dim = self.num_components * self.num_vars
        self.mean_head = nn.Linear(in_dim, out_dim)
        self.log_std_head = nn.Linear(in_dim, out_dim)
        self.logit_head = nn.Linear(in_dim, self.num_components)

        self.min_log_std = math.log(config.min_std)
        self.max_log_std = math.log(config.max_std)

    def _params(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(self.encoder(labels))
        batch = features.shape[0]
        shape = (batch, self.num_components, self.num_vars)

        logits = torch.log_softmax(self.logit_head(features), dim=-1)
        means = self.mean_head(features).view(shape)
        log_stds = (
            self.log_std_head(features)
            .view(shape)
            .clamp(min=self.min_log_std, max=self.max_log_std)
        )
        return logits, means, log_stds

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, means, log_stds = self._params(labels)

        normalized = (z.unsqueeze(1) - means) / log_stds.exp()
        component_log_prob = (
            -0.5 * (normalized**2 + math.log(2 * math.pi)) - log_stds
        ).sum(-1)

        return torch.logsumexp(logits + component_log_prob, dim=-1)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        logits, means, log_stds = self._params(labels)

        chosen = torch.distributions.Categorical(logits=logits).sample()
        index = chosen.view(-1, 1, 1).expand(-1, 1, self.num_vars)
        mean = means.gather(1, index).squeeze(1)
        log_std = log_stds.gather(1, index).squeeze(1)
        return mean + torch.randn_like(mean) * log_std.exp()

    def get_config(self) -> dict:
        return self.config.model_dump(mode="json")

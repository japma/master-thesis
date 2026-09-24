"""An unconditional PC over the latents: the CSPN's circuit without its hypernetwork."""

import copy
import math
from typing import Any

import torch
from networkx import DiGraph

from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from utils.config import SPNConfig


class SPN(AbstractCSPN):
    """Models `p(z)` with the parameters as ordinary weights. Implements the
    `AbstractCSPN` interface so it trains and samples wherever a CSPN does; the labels
    only set the batch size."""

    latent_mean: torch.Tensor
    latent_std: torch.Tensor

    def __init__(self, config: SPNConfig, graph: DiGraph[Any] | None = None) -> None:
        super().__init__()

        self.config = config

        if graph is not None:
            self.graph = graph
        else:
            self.graph = random_binary_trees(
                num_var=config.num_vars,
                depth=math.floor(math.log2(config.num_vars)),
                num_repetitions=config.num_repetitions,
            )

        self.topology_graph = copy.deepcopy(self.graph)

        self.args = Args(
            num_var=config.num_vars,
            num_dims=1,
            num_input_distributions=config.num_input_distributions,
            num_sums=config.num_sums,
            num_classes=1,
            exponential_family=NormalArray,
            exponential_family_args={
                "min_var": config.min_var,
                "max_var": config.max_var,
            },
        )

        self.einet = EinsumNetwork(graph=self.graph, param_nn=None, args=self.args)
        self.einet.initialize()

        if self.config.normalize_latents:
            self.register_buffer("latent_mean", torch.zeros(config.num_vars))
            self.register_buffer("latent_std", torch.ones(config.num_vars))

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Inject train-set latent mean/std (see dataset_loaders.latent_normalizer.
        LatentNormalizer.fit). Only valid when config.normalize_latents is True."""
        if not self.config.normalize_latents:
            raise RuntimeError(
                "Cannot set latent stats: config.normalize_latents is False"
            )
        with torch.no_grad():
            self.latent_mean.copy_(mean)
            self.latent_std.copy_(std)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_latents:
            return z
        return (z - self.latent_mean) / self.latent_std

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_latents:
            return z
        return z * self.latent_std + self.latent_mean

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Exact log p(z); `labels` is ignored."""
        log_prob = self.einet.forward(x=self._normalize(z)).squeeze(-1)
        if self.config.normalize_latents:
            log_prob = log_prob - self.latent_std.log().sum()
        return log_prob

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        samples = self.einet.sample(y=labels, std_correction=std_correction)
        assert samples is not None
        return self._denormalize(samples)

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_graph(self) -> DiGraph:
        return self.topology_graph

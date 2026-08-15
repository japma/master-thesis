import copy
import math
from typing import Any

import torch
from networkx import DiGraph

from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.psinet.conditioning_nn import build_conditioning_mlp_for
from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from models.cspn.psinet.label_encoder import (
    CategoricalLabelEncoder,
    LabelDropout,
    MultiBinaryLabelEncoder,
    MultiCategoricalLabelEncoder,
)
from utils.config import CSPNConfig, CSPNEncoderType


class PsiNetCSPN(AbstractCSPN):
    latent_mean: torch.Tensor
    latent_std: torch.Tensor

    def __init__(
        self,
        config: CSPNConfig,
        graph: DiGraph[Any] | None = None,
    ) -> None:
        """
        :param config: model configuration.
        :param graph: an already-constructed graph"""
        super().__init__()

        self.config = config

        if graph is not None:
            self.graph = graph
        else:
            depth = math.floor(math.log2(config.num_vars))
            self.graph = random_binary_trees(
                num_var=config.num_vars,
                depth=depth,
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

        self.einet = EinsumNetwork(
            graph=self.graph,
            param_nn=None,
            args=self.args,
        )
        self.einet.initialize()

        match config.encoder_config.encoder_type:
            case CSPNEncoderType.CATEGORICAL:
                encoder = CategoricalLabelEncoder(config.encoder_config.num_classes[0])
            case CSPNEncoderType.MULTI_BINARY:
                encoder = MultiBinaryLabelEncoder(config.encoder_config.num_classes[0])
            case CSPNEncoderType.MULTI_CATEGORICAL:
                encoder = MultiCategoricalLabelEncoder(
                    config.encoder_config.num_classes,
                )
            case _:
                raise ValueError("Illegal encoder type")

        self.label_dropout = LabelDropout(
            unknown_indices=encoder.unknown_indices,
            dropout_prob=0.15,  # TODO move to config
        )

        conditioning_network = build_conditioning_mlp_for(
            self.einet,
            h_dims=config.h_dims,
            encoder=encoder,
        )

        self.einet.param_nn = conditioning_network

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
        labels = self.label_dropout(labels)
        log_prob = self.einet.forward(x=self._normalize(z), y=labels).squeeze(-1)
        if self.config.normalize_latents:
            # change of variables for z_norm = (z - mean) / std:
            # log p(z) = log p_norm(z_norm) - sum(log std)
            log_prob = log_prob - self.latent_std.log().sum()
        return log_prob

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        samples = self.einet.sample(y=labels, std_correction=std_correction)
        assert samples is not None
        return self._denormalize(samples)

    def mpe(self, labels: torch.Tensor) -> torch.Tensor:
        mpe_samples = self.einet.mpe(y=labels)
        assert mpe_samples is not None
        return self._denormalize(mpe_samples)

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_graph(self) -> DiGraph:
        return self.topology_graph

import copy
import math
from collections.abc import Sequence
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

    def sample_conditional(
        self,
        labels: torch.Tensor,
        evidence: torch.Tensor,
        known_mask: torch.Tensor,
        std_correction: float = 1.0,
    ) -> torch.Tensor:
        """Sample p(z_unknown | z_known, labels), holding the known latent dims fixed.

        This is the tractable-inference path the whole point of using a PC rests on: the
        unknown dims are marginalized exactly rather than approximated, and the known ones
        come back unchanged.

        :param labels: conditioning labels, shape [B, ...] as accepted by the encoder.
        :param evidence: latent values, shape [B, num_vars]. Entries where `known_mask` is
                         False are ignored (they are marginalized), so they may hold any
                         placeholder.
        :param known_mask: boolean [B, num_vars], True for observed dims. Must be identical
                           for every row — marginalization is a property of the leaf layer,
                           not of an individual sample.
        :return: [B, num_vars] with the observed dims equal to `evidence`.
        """
        if evidence.shape != known_mask.shape:
            raise ValueError(
                f"evidence {tuple(evidence.shape)} and known_mask "
                f"{tuple(known_mask.shape)} must have the same shape"
            )
        if evidence.shape[0] != labels.shape[0]:
            raise ValueError(
                f"evidence batch {evidence.shape[0]} != labels batch {labels.shape[0]}"
            )
        if evidence.shape[1] != self.config.num_vars:
            raise ValueError(
                f"evidence has {evidence.shape[1]} dims, expected {self.config.num_vars}"
            )
        if not torch.all(known_mask == known_mask[0]):
            raise ValueError(
                "known_mask must be identical for every row in the batch; "
                "call sample_conditional() separately per distinct mask pattern."
            )
        if bool(known_mask[0].all()):
            raise ValueError("every dim is observed — nothing left to sample")

        unknown_idx: list[int] = (~known_mask[0]).nonzero(as_tuple=True)[0].tolist()

        # backtrack() runs its evidence forward pass through EinsumNetwork.forward, which
        # knows nothing about _normalize — so the evidence has to go in already normalized.
        # It pastes the observed dims straight back out of `x`, so _denormalize returns
        # them bit-for-bit.
        self.einet.set_marginalization_idx(unknown_idx)
        try:
            samples = self.einet.sample(
                x=self._normalize(evidence),
                y=labels,
                std_correction=std_correction,
            )
        finally:
            self.einet.set_marginalization_idx(None)

        assert samples is not None
        samples = self._denormalize(samples)

        # backtrack() pastes the observed dims back in *normalized* space, so with
        # normalize_latents on they come out of _denormalize a few ulps off what was
        # passed in. Restore the caller's values so "observed" means exactly that.
        known_idx = known_mask[0].nonzero(as_tuple=True)[0]
        samples[:, known_idx] = evidence[:, known_idx]
        return samples

    def sample_conditional_partial(
        self,
        labels: torch.Tensor,
        known: dict[int, float],
        std_correction: float = 1.0,
    ) -> torch.Tensor:
        """`sample_conditional` for the common case: a sparse {latent_idx: value} spec
        shared across the batch. Mirrors `LabelPC.complete_partial`."""
        if not known:
            raise ValueError(
                "`known` is empty — use sample() for unconditional sampling"
            )

        batch_size = labels.shape[0]
        device = labels.device
        evidence = torch.zeros(batch_size, self.config.num_vars, device=device)
        known_mask = torch.zeros(
            batch_size, self.config.num_vars, dtype=torch.bool, device=device
        )
        for idx, value in known.items():
            if not 0 <= idx < self.config.num_vars:
                raise ValueError(
                    f"latent index {idx} out of range [0, {self.config.num_vars})"
                )
            evidence[:, idx] = value
            known_mask[:, idx] = True

        return self.sample_conditional(
            labels, evidence, known_mask, std_correction=std_correction
        )

    def log_marginal(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        observed_idx: Sequence[int],
    ) -> torch.Tensor:
        """Exact log p(z_observed | labels), marginalizing every other latent dim.

        :param z: [B, num_vars]. Entries outside `observed_idx` are ignored.
        :param observed_idx: latent dims to evaluate. Order and duplicates don't matter.
        :return: [B] log-densities.
        """
        if z.shape[1] != self.config.num_vars:
            raise ValueError(
                f"z has {z.shape[1]} dims, expected {self.config.num_vars}"
            )
        observed = sorted(set(observed_idx))
        if not observed:
            raise ValueError("observed_idx is empty — the marginal would be constant 1")
        if observed[0] < 0 or observed[-1] >= self.config.num_vars:
            raise ValueError(
                f"observed_idx out of range [0, {self.config.num_vars}): {observed}"
            )

        unknown_idx = [i for i in range(self.config.num_vars) if i not in set(observed)]

        self.einet.set_marginalization_idx(unknown_idx)
        try:
            log_prob = self.einet.forward(x=self._normalize(z), y=labels).squeeze(-1)
        finally:
            self.einet.set_marginalization_idx(None)

        if self.config.normalize_latents:
            # Unlike forward(), only the observed dims went through the change of
            # variables — the marginalized ones contribute no Jacobian term.
            log_prob = log_prob - self.latent_std[observed].log().sum()
        return log_prob

    def mpe(self, labels: torch.Tensor) -> torch.Tensor:
        mpe_samples = self.einet.mpe(y=labels)
        assert mpe_samples is not None
        return self._denormalize(mpe_samples)

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_graph(self) -> DiGraph:
        return self.topology_graph

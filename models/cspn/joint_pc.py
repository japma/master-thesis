"""A plain (unconditional) PC over latents *and* labels jointly."""

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from networkx import DiGraph

from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import (
    CategoricalArray,
    MixedFamilyArray,
    NormalArray,
)
from models.cspn.psinet.graph import random_binary_trees
from utils.config import JointPCConfig


class JointPC(AbstractCSPN):
    """Models `p(z, y)` in one circuit, with the labels as ordinary categorical
    variables rather than the input of a conditioning hypernetwork.

    The variable order is the `num_latents` latent dimensions followed by one variable
    per label factor, so a conditional query is just a choice of which indices to
    marginalize. Unlike a CSPN this model has a `p(y)`, which is what makes
    "digit 3, colour unspecified" an exact marginal rather than a substitute for one.

    Implements the `AbstractCSPN` interface so it drops into `CSPNObjective` and the
    generation probes wherever a CSPN would go.
    """

    latent_mean: torch.Tensor
    latent_std: torch.Tensor

    def __init__(
        self,
        config: JointPCConfig,
        graph: DiGraph[Any] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_latents = config.num_latents
        self.label_cardinalities = list(config.label_cardinalities)
        self.num_vars = self.num_latents + len(self.label_cardinalities)

        if graph is not None:
            self.graph = graph
        else:
            depth = max(1, math.floor(math.log2(self.num_vars)))
            self.graph = random_binary_trees(
                num_var=self.num_vars,
                depth=depth,
                num_repetitions=config.num_repetitions,
            )

        self.topology_graph = copy.deepcopy(self.graph)

        self.args = Args(
            num_var=self.num_vars,
            num_dims=1,
            num_input_distributions=config.num_input_distributions,
            num_sums=config.num_sums,
            num_classes=1,
            exponential_family=MixedFamilyArray,
            exponential_family_args={
                "blocks": [
                    (
                        self.num_latents,
                        NormalArray,
                        {"min_var": config.min_var, "max_var": config.max_var},
                    ),
                    *(
                        (1, CategoricalArray, {"K": cardinality})
                        for cardinality in self.label_cardinalities
                    ),
                ]
            },
        )

        self.einet = EinsumNetwork(graph=self.graph, param_nn=None, args=self.args)
        self.einet.initialize()

        if self.config.normalize_latents:
            self.register_buffer("latent_mean", torch.zeros(self.num_latents))
            self.register_buffer("latent_std", torch.ones(self.num_latents))

    # --- variable layout ---

    @property
    def latent_idx(self) -> list[int]:
        return list(range(self.num_latents))

    @property
    def label_idx(self) -> list[int]:
        return list(range(self.num_latents, self.num_vars))

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

    def _pack(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Latents and labels as the single variable vector the circuit expects."""
        if z.shape[1] != self.num_latents:
            raise ValueError(
                f"z has {z.shape[1]} dims, expected {self.num_latents}"
            )
        if labels.shape[1] != len(self.label_cardinalities):
            raise ValueError(
                f"labels have {labels.shape[1]} factors, expected "
                f"{len(self.label_cardinalities)}"
            )
        return torch.cat([self._normalize(z), labels.float()], dim=1)

    def _log_jacobian(self, observed_latents: Sequence[int]) -> torch.Tensor | float:
        """Change-of-variables term for the normalized latent dims that were observed.
        Labels are discrete and marginalized dims contribute nothing."""
        if not self.config.normalize_latents or not observed_latents:
            return 0.0
        return self.latent_std[list(observed_latents)].log().sum()

    # --- density ---

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Exact log p(z, y)."""
        log_prob = self.einet.forward(x=self._pack(z, labels)).squeeze(-1)
        return log_prob - self._log_jacobian(self.latent_idx)

    def log_marginal(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        observed_idx: Sequence[int],
    ) -> torch.Tensor:
        """Exact log p(z_observed, y), marginalizing every other latent dim.

        :param observed_idx: latent dims to evaluate. Order and duplicates don't matter.
        """
        observed = sorted(set(observed_idx))
        if not observed:
            raise ValueError("observed_idx is empty — the marginal would be constant 1")
        if observed[0] < 0 or observed[-1] >= self.num_latents:
            raise ValueError(
                f"observed_idx out of range [0, {self.num_latents}): {observed}"
            )

        unknown = [i for i in self.latent_idx if i not in set(observed)]
        with self._marginalizing(unknown):
            log_prob = self.einet.forward(x=self._pack(z, labels)).squeeze(-1)
        return log_prob - self._log_jacobian(observed)

    def label_log_marginal(self, labels: torch.Tensor) -> torch.Tensor:
        """Exact log p(y), every latent dim marginalized out.

        The query a CSPN cannot answer at all: it has no p(y) to marginalize.
        """
        placeholder = torch.zeros(
            labels.shape[0], self.num_latents, device=labels.device
        )
        with self._marginalizing(self.latent_idx):
            return self.einet.forward(x=self._pack(placeholder, labels)).squeeze(-1)

    # --- sampling ---

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        """Sample p(z | y) with the labels held as evidence."""
        placeholder = torch.zeros(
            labels.shape[0], self.num_latents, device=labels.device
        )
        samples = self._backtrack(
            self._pack(placeholder, labels),
            marginalized=self.latent_idx,
            std_correction=std_correction,
        )
        return self._denormalize(samples[:, self.latent_idx])

    def sample_conditional(
        self,
        labels: torch.Tensor,
        evidence: torch.Tensor,
        known_mask: torch.Tensor,
        std_correction: float = 1.0,
    ) -> torch.Tensor:
        """Sample p(z_unknown | z_known, y), holding the known latent dims fixed.

        :param evidence: latent values, shape [B, num_latents]. Entries where
                         `known_mask` is False are ignored — they are marginalized.
        :param known_mask: boolean [B, num_latents], True for observed dims. Must be
                           identical for every row: marginalization is a property of
                           the leaf layer, not of an individual sample.
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
        if not torch.all(known_mask == known_mask[0]):
            raise ValueError(
                "known_mask must be identical for every row in the batch; "
                "call sample_conditional() separately per distinct mask pattern."
            )
        if bool(known_mask[0].all()):
            raise ValueError("every latent dim is observed — nothing left to sample")

        unknown = (~known_mask[0]).nonzero(as_tuple=True)[0].tolist()
        samples = self._backtrack(
            self._pack(evidence, labels),
            marginalized=unknown,
            std_correction=std_correction,
        )
        latents = self._denormalize(samples[:, self.latent_idx])

        # Backtracking pastes observed dims back in normalized space, so they return a
        # few ulps off. Restore the caller's values so "observed" means exactly that.
        known_idx = known_mask[0].nonzero(as_tuple=True)[0]
        latents[:, known_idx] = evidence[:, known_idx]
        return latents

    def sample_conditional_partial(
        self,
        labels: torch.Tensor,
        known: Mapping[int, float],
        std_correction: float = 1.0,
    ) -> torch.Tensor:
        """`sample_conditional` for a sparse {latent_idx: value} spec shared across the
        batch. Mirrors `PsiNetCSPN.sample_conditional_partial`."""
        if not known:
            raise ValueError("`known` is empty — use sample() for p(z | y)")

        batch_size = labels.shape[0]
        device = labels.device
        evidence = torch.zeros(batch_size, self.num_latents, device=device)
        known_mask = torch.zeros(
            batch_size, self.num_latents, dtype=torch.bool, device=device
        )
        for idx, value in known.items():
            if not 0 <= idx < self.num_latents:
                raise ValueError(
                    f"latent index {idx} out of range [0, {self.num_latents})"
                )
            evidence[:, idx] = value
            known_mask[:, idx] = True

        return self.sample_conditional(
            labels, evidence, known_mask, std_correction=std_correction
        )

    def sample_partial_labels(
        self,
        known: Mapping[int, int],
        batch_size: int,
        device: torch.device | None = None,
        std_correction: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample p(z, y_unspecified | y_specified) — "a 3, colour unspecified".

        The unspecified label factors are marginalized exactly and drawn from the
        model's own p(y), rather than filled in by a separate label model.

        :param known: label factor index (0-based over the label factors) -> value.
        :return: (latents, labels), the labels completed to a full factor vector.
        """
        num_factors = len(self.label_cardinalities)
        values = torch.zeros(batch_size, num_factors, device=device)
        specified: list[int] = []
        for factor, value in known.items():
            if not 0 <= factor < num_factors:
                raise ValueError(
                    f"label factor {factor} out of range [0, {num_factors})"
                )
            if not 0 <= value < self.label_cardinalities[factor]:
                raise ValueError(
                    f"value {value} out of range for label factor {factor} "
                    f"with {self.label_cardinalities[factor]} classes"
                )
            values[:, factor] = value
            specified.append(factor)

        marginalized = self.latent_idx + [
            self.num_latents + f for f in range(num_factors) if f not in set(specified)
        ]
        placeholder = torch.zeros(batch_size, self.num_latents, device=device)
        samples = self._backtrack(
            self._pack(placeholder, values),
            marginalized=marginalized,
            std_correction=std_correction,
        )
        latents = self._denormalize(samples[:, self.latent_idx])
        return latents, samples[:, self.label_idx].long()

    def mpe(self, labels: torch.Tensor) -> torch.Tensor:
        placeholder = torch.zeros(
            labels.shape[0], self.num_latents, device=labels.device
        )
        with self._marginalizing(self.latent_idx):
            samples = self.einet.mpe(x=self._pack(placeholder, labels))
        assert samples is not None
        return self._denormalize(samples[:, self.latent_idx])

    # --- plumbing ---

    class _Marginalizing:
        def __init__(self, einet: EinsumNetwork, idx: Sequence[int]) -> None:
            self.einet = einet
            self.idx = list(idx)

        def __enter__(self) -> None:
            self.einet.set_marginalization_idx(self.idx)

        def __exit__(self, *_: object) -> None:
            self.einet.set_marginalization_idx(None)

    def _marginalizing(self, idx: Sequence[int]) -> "JointPC._Marginalizing":
        return JointPC._Marginalizing(self.einet, idx)

    def _backtrack(
        self,
        packed: torch.Tensor,
        marginalized: Sequence[int],
        std_correction: float,
    ) -> torch.Tensor:
        with self._marginalizing(marginalized):
            samples = self.einet.sample(x=packed, std_correction=std_correction)
        assert samples is not None
        return samples

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_graph(self) -> DiGraph:
        return self.topology_graph

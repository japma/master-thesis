from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from networkx import DiGraph

from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import BinomialArray
from models.cspn.psinet.graph import random_binary_trees


class LabelPC(nn.Module):
    def __init__(
        self,
        num_attributes: int,
        num_input_distributions: int = 10,
        num_sums: int = 10,
        num_repetitions: int = 5,
        graph: DiGraph | None = None,
    ) -> None:
        super().__init__()
        self.num_attributes: int = num_attributes
        self.num_input_distributions: int = num_input_distributions
        self.num_sums: int = num_sums
        self.num_repetitions: int = num_repetitions

        if graph is not None:
            self.graph: DiGraph = graph
        else:
            depth: int = max(1, math.floor(math.log2(num_attributes)))
            self.graph = random_binary_trees(
                num_var=num_attributes, depth=depth, num_repetitions=num_repetitions
            )

        self.topology_graph: DiGraph = copy.deepcopy(self.graph)

        args = Args(
            num_var=num_attributes,
            num_dims=1,
            num_input_distributions=num_input_distributions,
            num_sums=num_sums,
            num_classes=1,
            exponential_family=BinomialArray,
            exponential_family_args={"N": 1},
        )

        self.einet = EinsumNetwork(graph=self.graph, param_nn=None, args=args)
        self.einet.initialize()

    def log_likelihood(self, attributes: torch.Tensor) -> torch.Tensor:
        return self.einet.forward(x=attributes.float()).squeeze(-1)

    @torch.no_grad()
    def complete(
        self, known_values: torch.Tensor, known_mask: torch.Tensor
    ) -> torch.Tensor:
        if not torch.all(known_mask == known_mask[0]):
            raise ValueError(
                "known_mask must be identical for every row in the batch; "
                "call complete() separately per distinct mask pattern."
            )
        unknown_idx: list[int] = (~known_mask[0]).nonzero(as_tuple=True)[0].tolist()

        self.einet.eval()
        self.einet.set_marginalization_idx(unknown_idx)
        try:
            completed = self.einet.sample(x=known_values.float())
        finally:
            self.einet.set_marginalization_idx(None)

        assert completed is not None
        return completed

    def complete_partial(
        self,
        known: dict[int, float],
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper around complete() for the common case: a sparse
        {attribute_idx: value} spec, shared across the whole batch, with every other
        attribute filled in by sampling from p(unknown attributes | known attributes)
        rather than left at an arbitrary placeholder value.

        :param known: attribute index -> value, for the attributes that are fixed.
                      Attributes not present in this dict are marginalized and sampled.
        :param batch_size: how many (independent) completions to draw.
        :return: (batch_size, num_attributes) tensor of concrete attribute values.
        """
        known_values = torch.zeros(batch_size, self.num_attributes, device=device)
        known_mask = torch.zeros(
            batch_size, self.num_attributes, dtype=torch.bool, device=device
        )
        for idx, value in known.items():
            known_values[:, idx] = value
            known_mask[:, idx] = True
        return self.complete(known_values, known_mask)

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(attributes)

    def get_config(self) -> dict:
        return {
            "num_attributes": self.num_attributes,
            "num_input_distributions": self.num_input_distributions,
            "num_sums": self.num_sums,
            "num_repetitions": self.num_repetitions,
        }

    def get_graph(self) -> DiGraph:
        return self.topology_graph

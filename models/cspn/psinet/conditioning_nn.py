from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cspn.psinet.factorized_leaf_layer import FactorizedLeafLayer
from models.cspn.psinet.sum_layer import EinsumLayer, EinsumMixingLayer


@dataclass(frozen=True)
class LabelSpec:
    kind: Literal["categorical", "multi_binary", "multi_categorical"]
    num_classes: int | None = None
    num_attributes: int | None = None
    cardinalities: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.kind == "categorical" and self.num_classes is None:
            raise AssertionError("categorical LabelSpec requires num_classes.")
        if self.kind == "multi_binary" and self.num_attributes is None:
            raise AssertionError("multi_binary LabelSpec requires num_attributes.")
        if self.kind == "multi_categorical" and not self.cardinalities:
            raise AssertionError("multi_categorical LabelSpec requires cardinalities.")

    @property
    def encoded_dim(self) -> int:
        if self.kind == "categorical":
            assert self.num_classes is not None
            return self.num_classes
        if self.kind == "multi_binary":
            assert self.num_attributes is not None
            return self.num_attributes
        assert self.cardinalities is not None
        return sum(self.cardinalities)

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        if self.kind == "categorical":
            if labels.dim() != 1:
                raise AssertionError(
                    f"Expected labels of shape (batch,), got {tuple(labels.shape)}."
                )
            return F.one_hot(labels, num_classes=self.num_classes).float()

        if self.kind == "multi_binary":
            if labels.dim() != 2 or labels.shape[1] != self.num_attributes:
                raise AssertionError(
                    f"Expected labels of shape (batch, {self.num_attributes}), "
                    f"got {tuple(labels.shape)}."
                )
            return labels.float()

        # multi_categorical
        assert self.cardinalities is not None
        if labels.dim() != 2 or labels.shape[1] != len(self.cardinalities):
            raise AssertionError(
                f"Expected labels of shape (batch, {len(self.cardinalities)}), "
                f"got {tuple(labels.shape)}."
            )
        parts = [
            F.one_hot(labels[:, i].long(), num_classes=card).float()
            for i, card in enumerate(self.cardinalities)
        ]
        return torch.cat(parts, dim=-1)


def derive_head_shapes(einet_layers: list) -> list[tuple[int, ...]]:
    if not isinstance(einet_layers[0], FactorizedLeafLayer):
        raise AssertionError("einet_layers[0] must be the FactorizedLeafLayer.")

    leaf = einet_layers[0]
    full_leaf_shape = leaf.ef_array.params_shape
    num_dims = leaf.num_dims
    if full_leaf_shape[-1] != 2 * num_dims:
        raise AssertionError(
            f"Expected leaf params_shape last dim == 2*num_dims ({2 * num_dims}), "
            f"got {full_leaf_shape[-1]}."
        )
    half_shape = (*full_leaf_shape[:-1], num_dims)

    shapes: list[tuple[int, ...]] = [half_shape, half_shape]  # mu head, var head

    for layer in einet_layers[1:]:
        if isinstance(layer, (EinsumLayer, EinsumMixingLayer)):
            shapes.append(tuple(layer.params_shape))
        else:
            raise AssertionError(
                f"Unexpected layer type in einet_layers[1:]: {type(layer)}"
            )

    return shapes


class ConditioningMLP(nn.Module):
    def __init__(
        self, label_spec: LabelSpec, einet_layers: list, h_dims: list[int]
    ) -> None:
        super().__init__()
        self.label_spec = label_spec

        head_shapes = derive_head_shapes(einet_layers)
        self.head_shapes = head_shapes
        self.num_layers = len(einet_layers)

        flat_sizes = [int(np.prod(s)) for s in head_shapes]

        dims = [label_spec.encoded_dim, *h_dims]
        self.trunk = nn.ModuleList(
            [nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(h_dims[-1], flat_sizes[i]) for i in range(len(head_shapes))]
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self, labels: torch.Tensor, x_unused: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        h = self.label_spec.encode(labels)
        for linear in self.trunk:
            h = linear(h)
            h = torch.relu(h)
            h = self.dropout(h)

        head_out = [head(h) for head in self.heads]

        reshaped = [
            o.reshape(o.shape[0], *shape)
            for o, shape in zip(head_out, self.head_shapes, strict=False)
        ]

        leaf_params = torch.cat(reshaped[:2], dim=-1)
        params = [leaf_params, *reshaped[2:]]
        return params


def build_conditioning_mlp_for(
    einet, label_spec: LabelSpec, h_dims: list[int]
) -> ConditioningMLP:
    return ConditioningMLP(
        label_spec=label_spec,
        einet_layers=einet.einet_layers,
        h_dims=h_dims,
    )

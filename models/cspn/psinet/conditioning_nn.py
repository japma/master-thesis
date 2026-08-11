from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.cspn.psinet.factorized_leaf_layer import FactorizedLeafLayer
from models.cspn.psinet.label_encoder import LabelEncoder
from models.cspn.psinet.param_shapes import derive_layer_param_shapes
from models.cspn.psinet.sum_layer import EinsumLayer, EinsumMixingLayer


def derive_head_shapes(einet_layers: list) -> list[tuple[int, ...]]:
    layer_shapes = derive_layer_param_shapes(einet_layers)

    leaf = einet_layers[0]
    full_leaf_shape = layer_shapes[0]
    num_dims = leaf.num_dims
    if full_leaf_shape[-1] != 2 * num_dims:
        raise AssertionError(
            f"Expected leaf params_shape last dim == 2*num_dims ({2 * num_dims}), "
            f"got {full_leaf_shape[-1]}."
        )
    half_shape = (*full_leaf_shape[:-1], num_dims)

    return [half_shape, half_shape, *layer_shapes[1:]]  # mu head, var head, sum-layer


class ConditioningMLP(nn.Module):
    def __init__(
        self, einet_layers: list, h_dims: list[int], encoder: LabelEncoder
    ) -> None:
        super().__init__()
        self.encoder = encoder
        head_shapes = derive_head_shapes(einet_layers)
        self.head_shapes = head_shapes
        self.num_layers = len(einet_layers)

        flat_sizes = [int(np.prod(s)) for s in head_shapes]

        dims = [encoder.num_classes, *h_dims]
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
        h = self.encoder(labels)
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
    einet,
    h_dims: list[int],
    encoder: LabelEncoder,
) -> ConditioningMLP:
    return ConditioningMLP(
        einet_layers=einet.einet_layers,
        h_dims=h_dims,
        encoder=encoder,
    )

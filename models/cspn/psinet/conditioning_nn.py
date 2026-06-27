import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cspn.psinet.factorized_leaf_layer import FactorizedLeafLayer
from models.cspn.psinet.sum_layer import EinsumLayer, EinsumMixingLayer


def derive_head_shapes(einet_layers) -> list[tuple]:
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

    shapes = [half_shape, half_shape]  # mu head, var head

    for layer in einet_layers[1:]:
        if isinstance(layer, (EinsumLayer, EinsumMixingLayer)):
            shapes.append(tuple(layer.params_shape))
        else:
            raise AssertionError(
                f"Unexpected layer type in einet_layers[1:]: {type(layer)}"
            )

    return shapes


class ConditioningMLP(nn.Module):
    def __init__(self, num_classes: int, einet_layers, h_dims: list[int]):
        super().__init__()
        self.num_classes = num_classes

        head_shapes = derive_head_shapes(einet_layers)
        self.head_shapes = head_shapes
        self.num_layers = len(einet_layers)

        flat_sizes = [int(np.prod(s)) for s in head_shapes]

        dims = [num_classes, *h_dims]
        self.trunk = nn.ModuleList(
            [nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(h_dims[-1], flat_sizes[i]) for i in range(len(head_shapes))]
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, labels: torch.Tensor, x_unused=None) -> list[torch.Tensor]:
        if labels.dim() != 1:
            raise AssertionError(
                f"Expected labels of shape (batch,), got shape {tuple(labels.shape)}."
            )

        h = F.one_hot(labels, num_classes=self.num_classes).float()
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
    einet, num_classes: int, h_dims: list[int]
) -> ConditioningMLP:
    return ConditioningMLP(
        num_classes=num_classes,
        einet_layers=einet.einet_layers,
        h_dims=h_dims,
    )

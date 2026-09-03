from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from models.cspn.psinet.label_encoder import LabelEncoder
from models.cspn.psinet.param_shapes import derive_layer_param_shapes
from utils.config import ConditioningType


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


class ConditioningNetwork(nn.Module, ABC):
    """Shared plumbing for hypernetworks that emit a full set of SPN parameters.

    Subclasses differ only in how they turn a label into the hidden vector `h`; the
    heads that turn `h` into per-layer parameter tensors are identical, and are
    deliberately a *single* linear map each. That linearity is what lets
    `FactorizedConditioningMLP` guarantee additivity in parameter space (see there).
    """

    def __init__(
        self, einet_layers: list, h_dims: list[int], encoder: LabelEncoder
    ) -> None:
        super().__init__()
        self.encoder = encoder
        head_shapes = derive_head_shapes(einet_layers)
        self.head_shapes = head_shapes
        self.num_layers = len(einet_layers)

        flat_sizes = [int(np.prod(s)) for s in head_shapes]
        self.heads = nn.ModuleList(
            [nn.Linear(h_dims[-1], flat_sizes[i]) for i in range(len(head_shapes))]
        )
        self.dropout = nn.Dropout(p=0.1)

    @abstractmethod
    def _hidden(self, labels: torch.Tensor) -> torch.Tensor:
        """Map raw labels to the pre-head hidden representation, shape [B, h_dims[-1]]."""
        ...

    def _assemble(self, h: torch.Tensor) -> list[torch.Tensor]:
        head_out = [head(h) for head in self.heads]

        reshaped = [
            o.reshape(o.shape[0], *shape)
            for o, shape in zip(head_out, self.head_shapes, strict=False)
        ]

        # heads[0] and heads[1] are the leaf mu and var halves; the leaf layer wants
        # them as one tensor.
        leaf_params = torch.cat(reshaped[:2], dim=-1)
        return [leaf_params, *reshaped[2:]]

    def forward(
        self, labels: torch.Tensor, x_unused: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        return self._assemble(self._hidden(labels))


def _linear_stack(in_dim: int, h_dims: list[int]) -> nn.ModuleList:
    """Bare Linears, activations applied by the caller.

    Only `ConditioningMLP` uses this, and only because its `trunk.<i>.weight` keys are
    baked into every existing checkpoint — folding the activations in here would
    renumber them. New variants should use `_trunk` instead.
    """
    dims = [in_dim, *h_dims]
    return nn.ModuleList([nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))])


def _trunk(in_dim: int, h_dims: list[int], dropout_p: float = 0.1) -> nn.Sequential:
    dims = [in_dim, *h_dims]
    layers: list[nn.Module] = []
    for i in range(1, len(dims)):
        layers += [nn.Linear(dims[i - 1], dims[i]), nn.ReLU(), nn.Dropout(p=dropout_p)]
    return nn.Sequential(*layers)


class ConditioningMLP(ConditioningNetwork):
    """One trunk over the whole encoded label.

    Every emitted parameter may depend on every label factor jointly. This is the
    original behaviour and stays the default; the attribute names (`trunk`, `heads`)
    are load-bearing for existing checkpoints.
    """

    def __init__(
        self, einet_layers: list, h_dims: list[int], encoder: LabelEncoder
    ) -> None:
        super().__init__(einet_layers, h_dims, encoder)
        self.trunk = _linear_stack(encoder.num_classes, h_dims)

    def _hidden(self, labels: torch.Tensor) -> torch.Tensor:
        h = self.encoder(labels)
        for linear in self.trunk:
            h = linear(h)
            h = torch.relu(h)
            h = self.dropout(h)
        return h


class FactorizedConditioningMLP(ConditioningNetwork):
    """One trunk per label factor, averaged before the heads.

    The point is compositional generalization. With a joint trunk, the parameters
    governing (say) background colour are a function of the whole label, so a
    (digit, background) pair absent from training has no reason to produce sane
    background parameters. Here each factor gets its own trunk and the contributions
    are combined additively, so a factor value is learned from every example
    containing it regardless of what it co-occurred with.

    Additivity survives all the way to the SPN parameters: the heads are single
    `nn.Linear` maps, so

        head(mean_a f_a(c_a)) = mean_a head_W(f_a(c_a)) + head_b

    i.e. the emitted parameters are an affine-additive function of the per-factor
    contributions, exactly. The per-factor trunks themselves stay nonlinear.

    The combination is a mean rather than a sum purely so the pre-head activation
    scale matches `ConditioningMLP` regardless of factor count (CelebA has 40), which
    keeps a joint-vs-factorized ablation on the same learning rate. The two differ
    only by a constant the heads can absorb.
    """

    def __init__(
        self, einet_layers: list, h_dims: list[int], encoder: LabelEncoder
    ) -> None:
        super().__init__(einet_layers, h_dims, encoder)

        factor_sizes = encoder.factor_sizes
        if len(factor_sizes) < 2:
            raise ValueError(
                f"{type(encoder).__name__} exposes {len(factor_sizes)} factor(s); "
                "factorized conditioning needs at least 2, otherwise it is identical "
                "to ConditioningMLP."
            )
        if sum(factor_sizes) != encoder.num_classes:
            raise AssertionError(
                f"{type(encoder).__name__}.factor_sizes sums to {sum(factor_sizes)} but "
                f"num_classes is {encoder.num_classes}; the slices would not tile the "
                "encoded label."
            )

        self.factor_sizes = factor_sizes
        self.trunks = nn.ModuleList([_trunk(size, h_dims) for size in factor_sizes])

    def _hidden(self, labels: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(labels)
        parts = torch.split(encoded, self.factor_sizes, dim=-1)

        stacked = torch.stack(
            [trunk(part) for part, trunk in zip(parts, self.trunks, strict=True)]
        )
        return stacked.mean(dim=0)


def build_conditioning_mlp_for(
    einet,
    h_dims: list[int],
    encoder: LabelEncoder,
    conditioning_type: ConditioningType = ConditioningType.JOINT,
) -> ConditioningNetwork:
    match conditioning_type:
        case ConditioningType.JOINT:
            cls: type[ConditioningNetwork] = ConditioningMLP
        case ConditioningType.FACTORIZED:
            cls = FactorizedConditioningMLP
        case _:
            raise ValueError(f"Unknown conditioning type {conditioning_type}")

    return cls(
        einet_layers=einet.einet_layers,
        h_dims=h_dims,
        encoder=encoder,
    )

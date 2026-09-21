from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F

from utils.config import CSPNEncoderConfig


class LabelEncoder(torch.nn.Module, ABC):
    """One-hot blocks, one per independently-varying label factor.

    With `allow_unknown`, every factor gets one extra slot past its real values, and
    `unknown_indices` names it. That slot is what "don't care" means to the
    hypernetwork: an ordinary input value it can learn a representation for, standing
    in for a marginal it cannot compute. Off by default, because turning it on widens
    the conditioning input and so invalidates every existing checkpoint.
    """

    def __init__(self, cardinalities: list[int], allow_unknown: bool = False) -> None:
        super().__init__()
        self._cardinalities: list[int] = list(cardinalities)
        self._allow_unknown: bool = allow_unknown

    @property
    def allow_unknown(self) -> bool:
        return self._allow_unknown

    @property
    def factor_sizes(self) -> list[int]:
        """Width of each factor in the encoded vector.

        `forward` concatenates one block per factor, so these are the slice widths of
        the output and must sum to `num_classes`. A conditioning network that wants to
        treat factors separately (see FactorizedConditioningMLP) slices on these.
        """
        extra = 1 if self._allow_unknown else 0
        return [c + extra for c in self._cardinalities]

    @property
    def num_classes(self) -> int:
        return sum(self.factor_sizes)

    @property
    def unknown_indices(self) -> list[int]:
        """The "no value given" index per factor: the slot past that factor's real
        values. Empty when the encoder has no unknown slot."""
        if not self._allow_unknown:
            return []
        return list(self._cardinalities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, factors)` indices to concatenated one-hot blocks."""
        parts = [
            F.one_hot(x[:, i], num_classes=size).float()
            for i, size in enumerate(self.factor_sizes)
        ]
        return torch.cat(parts, dim=-1)


class CategoricalLabelEncoder(LabelEncoder):
    """A single categorical factor, given as `(B,)` rather than `(B, 1)`."""

    def __init__(self, num_classes: int, allow_unknown: bool = False) -> None:
        super().__init__([num_classes], allow_unknown)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x, num_classes=self.factor_sizes[0]).float()


class MultiBinaryLabelEncoder(LabelEncoder):
    """Each attribute is a binary category: 0, 1."""

    def __init__(self, num_classes: int, allow_unknown: bool = False) -> None:
        super().__init__([2] * num_classes, allow_unknown)


class MultiCategoricalLabelEncoder(LabelEncoder):
    def __init__(self, cardinalities: list[int], allow_unknown: bool = False) -> None:
        super().__init__(cardinalities, allow_unknown)


class LabelDropout(torch.nn.Module):
    """Randomly replaces attribute labels with their 'unknown' index during training.
    Respects self.training automatically, same as nn.Dropout.
    """

    def __init__(self, unknown_indices: list[int], dropout_prob: float = 0.15) -> None:
        super().__init__()
        self.register_buffer(
            "unknown_indices", torch.tensor(unknown_indices, dtype=torch.long)
        )
        self.dropout_prob: float = dropout_prob

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout_prob <= 0.0:
            return labels

        is_multi: bool = labels.dim() == 2
        labels_2d: torch.Tensor = labels if is_multi else labels.unsqueeze(-1)

        drop_mask: torch.Tensor = (
            torch.rand_like(labels_2d, dtype=torch.float32) < self.dropout_prob
        )
        unknown_broadcast: torch.Tensor = (
            self.unknown_indices.to(labels.device).unsqueeze(0).expand_as(labels_2d)
        )
        labels_out: torch.Tensor = torch.where(drop_mask, unknown_broadcast, labels_2d)

        return labels_out if is_multi else labels_out.squeeze(-1)


def build_label_encoder(config: CSPNEncoderConfig) -> LabelEncoder:
    from utils.config import CSPNEncoderType

    allow_unknown = config.label_dropout_prob > 0.0
    match config.encoder_type:
        case CSPNEncoderType.CATEGORICAL:
            return CategoricalLabelEncoder(config.num_classes[0], allow_unknown)
        case CSPNEncoderType.MULTI_BINARY:
            return MultiBinaryLabelEncoder(config.num_classes[0], allow_unknown)
        case CSPNEncoderType.MULTI_CATEGORICAL:
            return MultiCategoricalLabelEncoder(config.num_classes, allow_unknown)
        case _:
            raise ValueError(f"Illegal encoder type {config.encoder_type!r}")

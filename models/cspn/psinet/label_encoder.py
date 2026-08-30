from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class LabelEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def factor_sizes(self) -> list[int]:
        """Width of each independently-varying factor in the encoded vector.

        `forward` concatenates one block per factor, so these are the slice widths of
        the output and must sum to `num_classes`. A conditioning network that wants to
        treat factors separately (see FactorizedConditioningMLP) slices on these.
        """
        ...


class CategoricalLabelEncoder(LabelEncoder):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_real_classes: int = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_real_classes

    @property
    def factor_sizes(self) -> list[int]:
        return [self.num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x, num_classes=self.num_classes).float()


class MultiBinaryLabelEncoder(LabelEncoder):
    """Each attribute is a binary category: 0, 1."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_attrs: int = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_attrs * 2

    @property
    def factor_sizes(self) -> list[int]:
        return [2] * self._num_attrs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            F.one_hot(x[:, i], num_classes=2).float() for i in range(self._num_attrs)
        ]
        return torch.cat(parts, dim=-1)


class MultiCategoricalLabelEncoder(LabelEncoder):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._real_cardinalities: list[int] = cardinalities

    @property
    def num_classes(self) -> int:
        return sum(c for c in self._real_cardinalities)

    @property
    def factor_sizes(self) -> list[int]:
        return list(self._real_cardinalities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            F.one_hot(x[:, i], num_classes=c).float()
            for i, c in enumerate(self._real_cardinalities)
        ]
        return torch.cat(parts, dim=-1)


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

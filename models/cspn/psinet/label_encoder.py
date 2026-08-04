from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class LabelEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass


class CategoricalLabelEncoder(LabelEncoder):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x, num_classes=self.num_classes).float()


class MultiBinaryLabelEncoder(LabelEncoder):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class MultiCategoricalLabelEncoder(LabelEncoder):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities

    @property
    def num_classes(self) -> int:
        return sum(self.cardinalities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            F.one_hot(x[:, i], num_classes=cardinality).float()
            for i, cardinality in enumerate(self.cardinalities)
        ]
        return torch.cat(parts, dim=-1)

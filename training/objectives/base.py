from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class StepOutput:
    metrics: dict[str, torch.Tensor]
    batch_size: int


class AbstractObjective(ABC):
    @abstractmethod
    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def val_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self) -> None:
        pass

    @abstractmethod
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

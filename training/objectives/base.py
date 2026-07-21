from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class StepOutput:
    metrics: dict[str, torch.Tensor]
    batch_size: int


class AbstractObjective(ABC):
    @abstractmethod
    def train_step(self, images: torch.Tensor) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def val_step(self, images: torch.Tensor) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self) -> None:
        pass

    @abstractmethod
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

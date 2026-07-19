from abc import ABC, abstractmethod

import torch


class StepOutput:
    metrics: dict
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

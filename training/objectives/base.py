from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from utils.checkpoints import (
    load_train_state as _load_train_state_dict,
)
from utils.checkpoints import (
    restore_train_state as _restore_train_state,
)
from utils.checkpoints import (
    save_train_state as _save_train_state_dict,
)


@dataclass
class StepOutput:
    metrics: dict[str, torch.Tensor]
    batch_size: int


class AbstractObjective(ABC):
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

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

    def extra_train_state(self) -> dict:
        """Override to persist objective-specific resume state, e.g. a beta
        annealing scheduler's step count."""
        return {}

    def load_extra_train_state(self, extra: dict) -> None:
        """Inverse of extra_train_state(); override alongside it."""
        return None

    def save_train_state(self, path: Path, epoch: int) -> None:
        """Saves optimizer/scheduler/RNG state for resuming a crashed run. Purely
        additive alongside save_checkpoint()'s model weights -- see
        utils.checkpoints.train_state_path."""
        _save_train_state_dict(
            path=path,
            epoch=epoch,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            extra=self.extra_train_state(),
        )

    def load_train_state(self, path: Path, device: torch.device | None = None) -> int:
        """Returns the epoch to resume from, or 0 if no train-state sidecar exists
        at `path` (e.g. a fresh run, or a checkpoint saved before this existed)."""
        state = _load_train_state_dict(path, device=device)
        if state is None:
            return 0
        start_epoch = _restore_train_state(state, self.optimizer, self.lr_scheduler)
        self.load_extra_train_state(state.get("extra", {}))
        return start_epoch

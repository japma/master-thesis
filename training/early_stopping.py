"""Early stopping with best-checkpoint saving."""

import copy

import torch.nn as nn

from utils.config import CSPNTrainingConfig


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        min_epochs: int = 0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs

        self.best_loss: float = float("inf")
        self.best_weights: dict | None = None
        self.best_epoch: int = 0
        self._counter: int = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Call once per epoch after validation.

        Returns True when training should stop, False otherwise. The best weights are
        tracked from the first epoch, but no stop is called before `min_epochs`.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience and epoch + 1 >= self.min_epochs

    @property
    def epochs_without_improvement(self) -> int:
        return self._counter

    def restore_best_weights(self, model: nn.Module) -> bool:
        """Load the best weights back into the model. False if there are none, which
        only happens when step() was never called."""
        if self.best_weights is None:
            return False
        model.load_state_dict(self.best_weights)
        return True


def build_early_stopping(training: CSPNTrainingConfig) -> EarlyStopping | None:
    """The stopper a latent-space run's config asks for; None when patience is unset."""
    if training.early_stopping_patience is None:
        print("Early stopping off")
        return None
    print(
        f"Early stopping on val total: patience {training.early_stopping_patience}, "
        f"min_delta {training.early_stopping_min_delta}, "
        f"not before epoch {training.early_stopping_min_epochs}"
    )
    return EarlyStopping(
        patience=training.early_stopping_patience,
        min_delta=training.early_stopping_min_delta,
        min_epochs=training.early_stopping_min_epochs,
    )

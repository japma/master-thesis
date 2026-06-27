"""Early stopping with best-checkpoint saving."""

import copy

import torch.nn as nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss: float = float("inf")
        self.best_weights: dict | None = None
        self.best_epoch: int = 0
        self._counter: int = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call once per epoch after validation.

        Returns True when training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = self._counter  # overwritten below with real epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load the best weights back into the model after training ends."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        else:
            raise RuntimeError("No best weights saved — was step() ever called?")

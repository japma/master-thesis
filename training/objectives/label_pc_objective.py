from __future__ import annotations

import torch

from models.cspn.psinet.label_pc import LabelPC
from training.losses.spn import NLLLoss
from training.objectives.base import AbstractObjective, StepOutput


class LabelPCObjective(AbstractObjective):
    """Objective for training the label-only PC over attribute vectors."""

    def __init__(
        self,
        model: LabelPC,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = NLLLoss()

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        if labels is None:
            raise ValueError("Labels must be provided for LabelPC training")

        self.model.train()
        outputs = self.model(labels.float())
        loss = self.loss_fn(outputs)

        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        metrics = {"total": loss.total}
        return StepOutput(metrics=metrics, batch_size=labels.size(0))

    @torch.no_grad()
    def val_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        if labels is None:
            raise ValueError("Labels must be provided for LabelPC training")

        self.model.eval()
        outputs = self.model(labels.float())
        loss = self.loss_fn(outputs)

        metrics = {"total": loss.total}
        return StepOutput(metrics=metrics, batch_size=labels.size(0))

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

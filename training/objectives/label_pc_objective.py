from __future__ import annotations

from pathlib import Path

import torch

from models.cspn.psinet.label_pc import LabelPC
from training.losses.spn import NLLLoss
from training.objectives.base import AbstractObjective, Batch, StepOutput
from utils.checkpoints import save_label_pc


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

    def train_step(self, batch: Batch) -> StepOutput:
        if batch.labels is None:
            raise ValueError("Labels must be provided for LabelPC training")
        labels = batch.labels

        self.model.train()
        outputs = self.model(labels.float())
        loss = self.loss_fn(outputs)

        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        metrics = {"total": loss.total}
        return StepOutput(metrics=metrics, batch_size=labels.size(0))

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        if batch.labels is None:
            raise ValueError("Labels must be provided for LabelPC training")
        labels = batch.labels

        self.model.eval()
        outputs = self.model(labels.float())
        loss = self.loss_fn(outputs)

        metrics = {"total": loss.total}
        return StepOutput(metrics=metrics, batch_size=labels.size(0))

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        save_label_pc(self.model, path)

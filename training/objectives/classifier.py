from pathlib import Path

import torch
import torch.nn as nn

from models.classifier import DigitClassifier
from training.metrics import PerClassAccuracy
from training.objectives.base import AbstractObjective, Batch, StepOutput
from utils.checkpoints import save_classifier
from utils.wandb_utils import log_metrics

# Colour-MNIST labels are [digit, fg, bg]; the judge reads only the first.
DIGIT_FACTOR = 0


class ClassifierObjective(AbstractObjective):
    """Cross-entropy on the digit factor of the label.

    Reports `error_rate` alongside `total` so early stopping, which minimises, keeps
    the most *accurate* epoch rather than the one with the lowest cross-entropy.
    """

    def __init__(
        self,
        model: DigitClassifier,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = PerClassAccuracy(model.config.num_classes)
        self._epoch = 0

    def train_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for classifier training"
            )
        images, digits = batch.images, batch.labels[:, DIGIT_FACTOR].long()

        self.model.train()
        logits = self.model(images)
        loss = self.loss_fn(logits, digits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        error_rate = (logits.detach().argmax(dim=1) != digits).float().mean()
        return StepOutput(
            metrics={"total": loss.detach(), "error_rate": error_rate},
            batch_size=images.size(0),
        )

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for classifier training"
            )
        images, digits = batch.images, batch.labels[:, DIGIT_FACTOR].long()

        self.model.eval()
        logits = self.model(images)
        loss = self.loss_fn(logits, digits)

        predictions = logits.argmax(dim=1)
        self.val_accuracy.update(predictions, digits)

        error_rate = (predictions != digits).float().mean()
        return StepOutput(
            metrics={"total": loss, "error_rate": error_rate},
            batch_size=images.size(0),
        )

    def on_epoch_end(self) -> None:
        per_digit = self.val_accuracy.per_class
        log_metrics(
            {
                f"val/digit_accuracy/{digit}": float(accuracy)
                for digit, accuracy in enumerate(per_digit)
            },
            step=self._epoch,
        )
        print(
            "Val accuracy per digit: "
            + "  ".join(f"{d}:{a:.3f}" for d, a in enumerate(per_digit))
        )
        self.val_accuracy.reset()
        self._epoch += 1
        self.lr_scheduler.step()

    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        save_classifier(self.model, path)

    def extra_train_state(self) -> dict:
        return {"epoch": self._epoch}

    def load_extra_train_state(self, extra: dict) -> None:
        self._epoch = int(extra.get("epoch", 0))

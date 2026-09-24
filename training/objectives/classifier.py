from pathlib import Path

import torch
import torch.nn as nn

from models.classifier import DigitClassifier
from training.metrics import PerClassAccuracy
from training.objectives.base import AbstractObjective, Batch, StepOutput
from utils.checkpoints import save_classifier
from utils.wandb_utils import log_metrics


class ClassifierObjective(AbstractObjective):
    """Summed cross-entropy over every label factor the classifier has a head for.

    Reports `error_rate/<factor>` alongside `total`, since accuracy is what the judge
    is for.
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
        self.names = model.config.names
        self.val_accuracy = [
            PerClassAccuracy(cardinality) for cardinality in model.config.cardinalities
        ]
        self._epoch = 0

    def _unpack(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for classifier training"
            )
        if batch.labels.shape[1] != len(self.names):
            raise ValueError(
                f"labels carry {batch.labels.shape[1]} factors but the classifier "
                f"has heads for {self.names}; drop `dataset.labels` from the config"
            )
        return batch.images, batch.labels.long()

    def _loss(self, logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return sum(
            (self.loss_fn(factor, labels[:, i]) for i, factor in enumerate(logits)),
            start=torch.zeros((), device=labels.device),
        )

    def _error_rates(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        wrong = (predictions != labels).float().mean(dim=0)
        return {f"error_rate/{name}": wrong[i] for i, name in enumerate(self.names)}

    def train_step(self, batch: Batch) -> StepOutput:
        images, labels = self._unpack(batch)

        self.model.train()
        logits = self.model(images)
        loss = self._loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        predictions = torch.stack([f.detach().argmax(dim=1) for f in logits], dim=1)
        return StepOutput(
            metrics={"total": loss.detach()} | self._error_rates(predictions, labels),
            batch_size=images.size(0),
        )

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        images, labels = self._unpack(batch)

        self.model.eval()
        logits = self.model(images)
        loss = self._loss(logits, labels)

        predictions = torch.stack([f.argmax(dim=1) for f in logits], dim=1)
        for i, accuracy in enumerate(self.val_accuracy):
            accuracy.update(predictions[:, i], labels[:, i])

        return StepOutput(
            metrics={"total": loss} | self._error_rates(predictions, labels),
            batch_size=images.size(0),
        )

    def on_epoch_end(self) -> None:
        for name, accuracy in zip(self.names, self.val_accuracy, strict=True):
            per_class = accuracy.per_class
            log_metrics(
                {
                    f"val/{name}_accuracy/{c}": float(value)
                    for c, value in enumerate(per_class)
                },
                step=self._epoch,
            )
            print(
                f"Val accuracy per {name}: "
                + "  ".join(f"{c}:{a:.3f}" for c, a in enumerate(per_class))
            )
            accuracy.reset()
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

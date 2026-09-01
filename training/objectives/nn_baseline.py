from pathlib import Path

import torch

from models.autoencoder import AbstractAutoencoder
from models.neural_baseline import AbstractNeuralBaseline
from training.losses.spn import NLLLoss
from training.objectives.base import AbstractObjective, Batch, StepOutput
from utils.checkpoints import save_nn_baseline


class NeuralBaselineObjective(AbstractObjective):
    def __init__(
        self,
        model: AbstractNeuralBaseline,
        autoencoder: AbstractAutoencoder,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = NLLLoss()

    def train_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError("Images and labels must be provided for baseline training")
        images, labels = batch.images, batch.labels

        self.model.train()
        with torch.no_grad():
            latent: torch.Tensor = self.autoencoder.encode(images)

        loss = self.loss_fn(self.model(latent, labels.long()))

        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        return StepOutput(metrics={"total": loss.total}, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError("Images and labels must be provided for baseline training")
        images, labels = batch.images, batch.labels

        self.model.eval()
        loss = self.loss_fn(self.model(self.autoencoder.encode(images), labels.long()))

        return StepOutput(metrics={"total": loss.total}, batch_size=images.size(0))

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.autoencoder.decode(self.model.sample(samples.long()))

    def save_checkpoint(self, path: Path) -> None:
        save_nn_baseline(self.model, path)

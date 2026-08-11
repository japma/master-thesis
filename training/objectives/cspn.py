from pathlib import Path

import torch

from models.autoencoder import AbstractAutoencoder
from models.cspn.abstract_cspn import AbstractCSPN
from training.losses.spn import NLLLoss
from training.objectives.base import AbstractObjective, StepOutput
from utils.checkpoints import save_cspn


class CSPNObjective(AbstractObjective):
    """
    Objective for CSPN
    """

    def __init__(
        self,
        model: AbstractCSPN,
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

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        """Step function used for training
        Args:
            images: torch.Tensor:
            labels:
        """
        self.model.train()
        with torch.no_grad():
            latent: torch.Tensor = self.autoencoder.encode(images)

        if labels is None:
            raise ValueError("Labels must be provided for CSPN training")
        outputs = self.model(latent, labels.long())
        loss = self.loss_fn(outputs)

        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        metrics = {
            "total": loss.total,
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        self.model.eval()
        with torch.no_grad():
            latent: torch.Tensor = self.autoencoder.encode(images)
            if labels is None:
                raise ValueError("Labels must be provided for CSPN training")
            outputs = self.model(latent, labels.long())
            loss = self.loss_fn(outputs)

        metrics = {
            "total": loss.total,
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            sampled_latent: torch.Tensor = self.model.sample(samples.long())
            sampled_images: torch.Tensor = self.autoencoder.decode(sampled_latent)
        return sampled_images

    def save_checkpoint(self, path: Path) -> None:
        # TODO maybe save optimizer and scheduler state as well
        save_cspn(self.model, path)

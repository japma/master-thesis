from typing import TypedDict

import torch

from models import VariationalAutoencoder
from training.losses import BetaVAELoss, VAELoss
from training.objectives.base import AbstractObjective, StepOutput


class BetaVAETrainMetrics(TypedDict):
    total: float
    recon: float
    kl: float
    perceptual: float


class BetaVAEObjective(AbstractObjective):
    """
    Objective for Beta VAE
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        device = next(model.parameters()).device
        self.loss_fn: VAELoss = VAELoss(beta_vae_loss=BetaVAELoss()).to(device)

    def train_step(self, images: torch.Tensor) -> StepOutput:
        """Step function used for training
        :param images
        """
        logits, mu, log_var, z = self.model(images)
        loss = self.loss_fn(images, logits, mu, log_var)
        self.optimizer.zero_grad()
        loss["total"].backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def val_step(self, images: torch.Tensor) -> StepOutput:
        logits, mu, log_var, z = self.model(images)
        loss = self.loss_fn(images, logits, mu, log_var)

        return loss

    def on_epoch_end(self) -> None:
        self.scheduler.step()

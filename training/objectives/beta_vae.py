import torch

from models import VariationalAutoencoder
from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.vae import VAELoss, VAELossOutput
from training.objectives.base import AbstractObjective, StepOutput
from training.schedulers import BetaAnnealingScheduler


class BetaVAEObjective(AbstractObjective):
    """
    Objective for Beta VAE
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: VAELoss,
        beta_scheduler: BetaAnnealingScheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: VAELoss = loss_fn
        self.beta_scheduler = beta_scheduler

    def train_step(self, images: torch.Tensor) -> StepOutput:
        """Step function used for training
        :param images
        """
        self.model.train()
        outputs: VAEForwardOutput = self.model(images)

        current_beta: float = self.beta_scheduler.beta
        loss: VAELossOutput = self.loss_fn(images, outputs, beta=current_beta)
        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()
        self.beta_scheduler.step()

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "beta": torch.tensor(current_beta),
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, images: torch.Tensor) -> StepOutput:
        self.model.eval()
        with torch.no_grad():
            outputs: VAEForwardOutput = self.model(images)
            loss: VAELossOutput = self.loss_fn(images, outputs)

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs: VAEForwardOutput = self.model(samples)
        return torch.sigmoid(outputs.reconstructed)

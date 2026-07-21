import torch

from models import VariationalAutoencoder
from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.vae import VAELoss, VAELossOutput
from training.objectives.base import AbstractObjective, StepOutput


class BetaVAEObjective(AbstractObjective):
    """
    Objective for Beta VAE
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: VAELoss,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn: VAELoss = loss_fn

    def train_step(self, images: torch.Tensor) -> StepOutput:
        """Step function used for training
        :param images
        """
        self.model.train()
        outputs: VAEForwardOutput = self.model(images)
        loss: VAELossOutput = self.loss_fn(images, outputs)
        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        metrics = {
            "total": loss.total.item(),
            "recon": loss.recon.item(),
            "kl": loss.kl.item(),
            "perceptual": loss.perceptual.item(),
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, images: torch.Tensor) -> StepOutput:
        self.model.eval()
        with torch.no_grad():
            outputs: VAEForwardOutput = self.model(images)
            loss: VAELossOutput = self.loss_fn(images, outputs)

        metrics = {
            "total": loss.total.item(),
            "recon": loss.recon.item(),
            "kl": loss.kl.item(),
            "perceptual": loss.perceptual.item(),
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def on_epoch_end(self) -> None:
        self.scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs: VAEForwardOutput = self.model(samples)
        return torch.sigmoid(outputs.reconstructed)

import torch

from models import VariationalAutoencoder
from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.tcvae import BetaTCVAELoss, TCVAELossOutput
from training.objectives.base import AbstractObjective, StepOutput


class TCVAEObjective(AbstractObjective):
    """
    Objective for TCVAE
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: BetaTCVAELoss,
        train_data_size: int,
        test_data_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: BetaTCVAELoss = loss_fn
        self.train_data_size = train_data_size
        self.test_data_size = test_data_size

    def train_step(self, images: torch.Tensor) -> StepOutput:
        """Step function used for training
        :param images
        """
        self.model.train()
        outputs: VAEForwardOutput = self.model(images)

        loss: TCVAELossOutput = self.loss_fn(
            images, outputs, dataset_size=self.train_data_size
        )
        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()

        metrics = {
            "total": loss.total.item(),
            "recon": loss.recon.item(),
            "kl": loss.kl.item(),
            # TODO maybe add the perc loss here as well
            # "perceptual": loss.perceptual.item(),
            "mi": loss.mi.item(),
            "tc": loss.tc.item(),
            "dwkl": loss.dwkl.item(),
        }
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, images: torch.Tensor) -> StepOutput:
        self.model.eval()
        with torch.no_grad():
            outputs: VAEForwardOutput = self.model(images)
            loss: TCVAELossOutput = self.loss_fn(
                images, outputs, dataset_size=self.test_data_size
            )

        metrics = {
            "total": loss.total.item(),
            "recon": loss.recon.item(),
            "kl": loss.kl.item(),
            # "perceptual": loss.perceptual.item(),
            "mi": loss.mi.item(),
            "tc": loss.tc.item(),
            "dwkl": loss.dwkl.item(),
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

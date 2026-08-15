from pathlib import Path

import torch

from models import VariationalAutoencoder
from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.tcvae import BetaTCVAELoss, TCVAELossOutput
from training.objectives.base import AbstractObjective, StepOutput
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import save_autoencoder


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
        beta_scheduler: BetaAnnealingScheduler,
        train_data_size: int,
        test_data_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: BetaTCVAELoss = loss_fn
        self.beta_scheduler = beta_scheduler
        self.train_data_size = train_data_size
        self.test_data_size = test_data_size

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        """Step function used for training
        :param images
        :param labels
        """
        self.model.train()
        outputs: VAEForwardOutput = self.model(images)

        current_beta: float = self.beta_scheduler.beta
        loss: TCVAELossOutput = self.loss_fn(
            images, outputs, dataset_size=self.train_data_size, beta=current_beta
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        self.optimizer.step()
        self.beta_scheduler.step()

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "mi": loss.mi,
            "tc": loss.tc,
            "dwkl": loss.dwkl,
            "beta": torch.tensor(current_beta),
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
            outputs: VAEForwardOutput = self.model(images)
            loss: TCVAELossOutput = self.loss_fn(
                images, outputs, dataset_size=self.test_data_size
            )

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "mi": loss.mi,
            "tc": loss.tc,
            "dwkl": loss.dwkl,
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

    def save_checkpoint(self, path: Path) -> None:
        save_autoencoder(self.model, path)
        print("Saved TCVAE checkpoint to", path)

    def extra_train_state(self) -> dict:
        return {"beta_scheduler_step": self.beta_scheduler.current_step}

    def load_extra_train_state(self, extra: dict) -> None:
        if "beta_scheduler_step" in extra:
            self.beta_scheduler.current_step = extra["beta_scheduler_step"]

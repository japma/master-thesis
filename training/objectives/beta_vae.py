from training.losses.base import kl_per_dimension
from pathlib import Path

import torch

from models import VariationalAutoencoder
from models.autoencoder.variational_autoencoder import VAEForwardOutput
from training.losses.vae import VAELoss, VAELossOutput
from training.objectives.base import AbstractObjective, StepOutput
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import save_autoencoder


class BetaVAEObjective(AbstractObjective):
    def __init__(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: VAELoss,
        beta_scheduler: BetaAnnealingScheduler,
        max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: VAELoss = loss_fn
        self.beta_scheduler = beta_scheduler
        self.max_grad_norm: float = max_grad_norm

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepOutput:
        self.model.train()
        outputs: VAEForwardOutput = self.model(images)

        current_beta: float = self.beta_scheduler.beta
        loss: VAELossOutput = self.loss_fn(images, outputs, beta=current_beta)
        self.optimizer.zero_grad()
        loss.total.backward()
        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        self.beta_scheduler.step()

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "beta": torch.tensor(current_beta),
            "grad_norm": grad_norm,
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
            loss: VAELossOutput = self.loss_fn(images, outputs)
            kl_dim: torch.Tensor = kl_per_dimension(outputs.mu, outputs.log_var)

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "kl_per_dim": kl_dim,
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
        print("Saved Beta VAE checkpoint to", path)

    def extra_train_state(self) -> dict:
        return {"beta_scheduler_step": self.beta_scheduler.current_step}

    def load_extra_train_state(self, extra: dict) -> None:
        if "beta_scheduler_step" in extra:
            self.beta_scheduler.current_step = extra["beta_scheduler_step"]

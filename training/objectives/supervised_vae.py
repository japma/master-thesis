from pathlib import Path

import torch

from models.autoencoder.supervised_vae import (
    SupervisedVAE,
    SupervisedVAEForwardOutput,
)
from training.losses.base import kl_per_dimension
from training.losses.supervised_vae import SupervisedVAELoss, SupervisedVAELossOutput
from training.objectives.base import AbstractObjective, Batch, StepOutput
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import save_autoencoder


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == targets).float().mean()


class SupervisedVAEObjective(AbstractObjective):
    """Beta-VAE training with the latent classification heads attached.

    `gamma_scheduler` ramps the classification weight the same way `beta_scheduler`
    ramps the KL one; validation always scores at the loss's configured final gamma so
    `total` stays comparable across epochs (and usable for early stopping).
    """

    def __init__(
        self,
        model: SupervisedVAE,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: SupervisedVAELoss,
        beta_scheduler: BetaAnnealingScheduler,
        gamma_scheduler: BetaAnnealingScheduler,
        factor_names: list[str],
        max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: SupervisedVAELoss = loss_fn
        self.beta_scheduler = beta_scheduler
        self.gamma_scheduler = gamma_scheduler
        self.factor_names = factor_names
        self.max_grad_norm: float = max_grad_norm

    def train_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for supervised VAE training"
            )
        images, labels = batch.images, batch.labels

        self.model.train()
        outputs: SupervisedVAEForwardOutput = self.model(images)

        current_beta: float = self.beta_scheduler.beta
        current_gamma: float = self.gamma_scheduler.beta
        loss: SupervisedVAELossOutput = self.loss_fn(
            images, outputs, labels, beta=current_beta, gamma=current_gamma
        )

        self.optimizer.zero_grad()
        loss.total.backward()
        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        self.beta_scheduler.step()
        self.gamma_scheduler.step()

        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "classification": loss.classification,
            "beta": torch.tensor(current_beta),
            "gamma": torch.tensor(current_gamma),
            "grad_norm": grad_norm,
        }
        metrics.update(self._per_factor_metrics(loss))
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for supervised VAE training"
            )
        images, labels = batch.images, batch.labels

        self.model.eval()
        outputs: SupervisedVAEForwardOutput = self.model(images)
        loss: SupervisedVAELossOutput = self.loss_fn(images, outputs, labels)
        kl_dim: torch.Tensor = kl_per_dimension(outputs.mu, outputs.log_var)

        targets = labels if labels.ndim > 1 else labels.unsqueeze(1)
        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "classification": loss.classification,
            "kl_per_dim": kl_dim,
        }
        metrics.update(self._per_factor_metrics(loss))
        for name, logits, target in zip(
            self.factor_names, outputs.logits, targets.T, strict=True
        ):
            metrics[f"acc/{name}"] = _accuracy(logits, target)
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def _per_factor_metrics(
        self, loss: SupervisedVAELossOutput
    ) -> dict[str, torch.Tensor]:
        return {
            f"ce/{name}": value
            for name, value in zip(self.factor_names, loss.per_factor, strict=True)
        }

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outputs: SupervisedVAEForwardOutput = self.model(samples)
        return torch.sigmoid(outputs.reconstructed)

    def save_checkpoint(self, path: Path) -> None:
        save_autoencoder(self.model, path)
        print("Saved supervised VAE checkpoint to", path)

    def extra_train_state(self) -> dict:
        return {
            "beta_scheduler_step": self.beta_scheduler.current_step,
            "gamma_scheduler_step": self.gamma_scheduler.current_step,
        }

    def load_extra_train_state(self, extra: dict) -> None:
        if "beta_scheduler_step" in extra:
            self.beta_scheduler.current_step = extra["beta_scheduler_step"]
        if "gamma_scheduler_step" in extra:
            self.gamma_scheduler.current_step = extra["gamma_scheduler_step"]

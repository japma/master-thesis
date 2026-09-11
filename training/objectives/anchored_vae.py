from pathlib import Path

import torch

from models.autoencoder.anchored_vae import AnchoredVAE, AnchoredVAEForwardOutput
from training.losses.anchored_vae import AnchoredVAELoss, AnchoredVAELossOutput
from training.losses.base import kl_per_dimension
from training.objectives.base import AbstractObjective, Batch, StepOutput
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import save_autoencoder


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == targets).float().mean()


class AnchoredVAEObjective(AbstractObjective):
    """Beta-VAE training with anchored label blocks.    """

    def __init__(
        self,
        model: AnchoredVAE,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: AnchoredVAELoss,
        beta_scheduler: BetaAnnealingScheduler,
        gamma_scheduler: BetaAnnealingScheduler,
        factor_names: list[str],
        max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn: AnchoredVAELoss = loss_fn
        self.beta_scheduler = beta_scheduler
        self.gamma_scheduler = gamma_scheduler
        self.factor_names = factor_names
        self.anchor_names = [factor_names[i] for i in model.anchored_factors]
        self.head_names = [factor_names[i] for i in model.head_factors]
        self.max_grad_norm: float = max_grad_norm

    def _require(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.images is None or batch.labels is None:
            raise ValueError(
                "Images and labels must be provided for anchored VAE training"
            )
        return batch.images, batch.labels

    def train_step(self, batch: Batch) -> StepOutput:
        images, labels = self._require(batch)

        self.model.train()
        outputs: AnchoredVAEForwardOutput = self.model(images)

        current_beta: float = self.beta_scheduler.beta
        current_gamma: float = self.gamma_scheduler.beta
        loss: AnchoredVAELossOutput = self.loss_fn(
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
            "anchor": loss.anchor,
            "classification": loss.classification,
            "beta": torch.tensor(current_beta),
            "gamma": torch.tensor(current_gamma),
            "grad_norm": grad_norm,
        }
        metrics.update(self._per_factor_metrics(loss))
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        images, labels = self._require(batch)

        self.model.eval()
        outputs: AnchoredVAEForwardOutput = self.model(images)
        loss: AnchoredVAELossOutput = self.loss_fn(images, outputs, labels)
        kl_dim: torch.Tensor = kl_per_dimension(outputs.mu, outputs.log_var)

        targets = labels if labels.ndim > 1 else labels.unsqueeze(1)
        metrics = {
            "total": loss.total,
            "recon": loss.recon,
            "kl": loss.kl,
            "perceptual": loss.perceptual,
            "anchor": loss.anchor,
            "classification": loss.classification,
            "kl_per_dim": kl_dim,
        }
        metrics.update(self._per_factor_metrics(loss))
        for name, rmse in zip(
            self.anchor_names,
            self.loss_fn.anchor_rmse(outputs.mu, targets),
            strict=True,
        ):
            metrics[f"anchor_rmse/{name}"] = rmse
        for name, logits, factor in zip(
            self.head_names, outputs.logits, self.model.head_factors, strict=True
        ):
            metrics[f"acc/{name}"] = _accuracy(logits, targets[:, factor])
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def _per_factor_metrics(
        self, loss: AnchoredVAELossOutput
    ) -> dict[str, torch.Tensor]:
        metrics = {
            f"anchor_kl/{name}": value
            for name, value in zip(self.anchor_names, loss.per_anchor, strict=True)
        }
        metrics.update(
            {
                f"ce/{name}": value
                for name, value in zip(self.head_names, loss.per_head, strict=True)
            }
        )
        return metrics

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()

    @torch.no_grad()
    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outputs: AnchoredVAEForwardOutput = self.model(samples)
        return torch.sigmoid(outputs.reconstructed)

    def save_checkpoint(self, path: Path) -> None:
        save_autoencoder(self.model, path)
        print("Saved anchored VAE checkpoint to", path)

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

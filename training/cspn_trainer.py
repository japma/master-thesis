"""CSPN training loop."""

import torch
import tqdm
from rtpt import RTPT

import wandb
from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective
from utils.config import CSPNRunConfig


def train_cspn(
    objective: AbstractObjective,
    device: torch.device,
    cfg: CSPNRunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
) -> None:
    epochs = cfg.training.epochs
    log_sample_every = 10
    sample_count = max(1, min(16, cfg.dataset.num_classes))

    train_metrics = MetricsCollector()
    val_metrics = MetricsCollector()

    for epoch in range(epochs):
        # Train step
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Train {epoch + 1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = objective.train_step(images, labels)
            train_metrics.update(loss)

        avg_train_loss = train_metrics.compute_average_metrics()
        print(f"Train Loss: {avg_train_loss}")

        for images, labels in tqdm.tqdm(test_loader, desc=f"Test {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = objective.val_step(images, labels)
            val_metrics.update(loss)

        avg_val_loss = val_metrics.compute_average_metrics()
        print(f"Val Loss: {avg_val_loss}")

        metrics = {}
        for key, value in avg_train_loss.items():
            metrics[f"train/{key}"] = value
        for key, value in avg_val_loss.items():
            metrics[f"val/{key}"] = value

        wandb.log(metrics, step=epoch)
        train_metrics.reset()
        val_metrics.reset()

        if epoch % log_sample_every == 0:
            sample_labels = torch.arange(sample_count, device=device, dtype=torch.long)
            samples = objective.sample(sample_labels)
            samples_u8 = (samples.clamp(0, 1) * 255).byte().cpu()
            wandb.log(
                {
                    "samples/cspn_generated_images": [
                        wandb.Image(img) for img in samples_u8
                    ]
                },
                step=epoch,
            )

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    # Final samples
    final_sample_labels = torch.arange(sample_count, device=device, dtype=torch.long)
    final_samples = objective.sample(final_sample_labels)
    final_samples_u8 = (final_samples.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {
            "samples/cspn_generated_images": [
                wandb.Image(img) for img in final_samples_u8
            ]
        },
        step=epoch,
    )

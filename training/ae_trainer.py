"""Autoencoder training loop."""

from collections.abc import Sized

import torch
import tqdm
from rtpt import RTPT

import wandb
from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective
from utils.config import AERunConfig


def train_autoencoder(
    objective: AbstractObjective,
    device: torch.device,
    cfg: AERunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
) -> None:
    epochs = cfg.training.epochs
    log_sample_every = 10

    dataset = test_loader.dataset

    if not isinstance(dataset, Sized):
        raise TypeError("Test dataset must be sized")

    dataset_len = len(dataset)

    sample_indices = torch.randperm(dataset_len)[: min(10, dataset_len)]
    sample_images = torch.stack([test_loader.dataset[i][0] for i in sample_indices]).to(
        device
    )

    sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/input": [wandb.Image(img) for img in sample_images_u8]},
        step=0,
    )

    train_metrics = MetricsCollector()
    val_metrics = MetricsCollector()

    for epoch in range(epochs):
        # Train step
        for images, _ in tqdm.tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            loss = objective.train_step(images)
            train_metrics.update(loss)

        avg_train_loss = train_metrics.compute_average_metrics()
        print(f"Loss: {avg_train_loss}")

        # Val step
        for images, _ in tqdm.tqdm(test_loader, desc=f"Test {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            loss = objective.val_step(images)
            val_metrics.update(loss)

        avg_val_loss = val_metrics.compute_average_metrics()
        print(f"Loss: {avg_val_loss}")

        metrics = {}
        for key, value in avg_train_loss.items():
            metrics[f"train/{key}"] = value
        for key, value in avg_val_loss.items():
            metrics[f"val/{key}"] = value

        wandb.log(metrics, step=epoch)

        train_metrics.reset()
        val_metrics.reset()

        if epoch % log_sample_every == 0:
            reconstructed_samples = objective.sample(sample_images)
            reconstructed_samples_u8 = (
                (reconstructed_samples.clamp(0, 1) * 255).byte().cpu()
            )
            wandb.log(
                {
                    "samples/reconstructed": [
                        wandb.Image(img) for img in reconstructed_samples_u8
                    ]
                },
                step=epoch,
            )

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    # Final reconstructions
    final_reconstructed_samples = objective.sample(sample_images)
    final_reconstructed_samples_u8 = (
        (final_reconstructed_samples.clamp(0, 1) * 255).byte().cpu()
    )
    wandb.log(
        {
            "samples/reconstructed": [
                wandb.Image(img) for img in final_reconstructed_samples_u8
            ]
        },
        step=epoch,
    )

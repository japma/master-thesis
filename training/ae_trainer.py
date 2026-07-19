"""Autoencoder training loop."""

import torch
import tqdm
from rtpt import RTPT

import wandb
from models.autoencoder import AbstractAutoencoder
from training.logging import log_generations, log_reconstructions
from training.losses import VAELoss, kl_per_dim
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
    warmup_epochs = cfg.training.kl_warmup_epochs
    log_sample_every = 10

    # pyrefly: ignore [bad-argument-type]
    sample_indices = torch.randperm(len(test_loader.dataset))[:16]
    sample_images = torch.stack([test_loader.dataset[i][0] for i in sample_indices]).to(
        device
    )

    sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/input": [wandb.Image(img) for img in sample_images_u8]},
        step=0,
    )

    for epoch in range(epochs):
        annealed_beta = min(1.0, epoch / max(warmup_epochs, 1))

        train_losses = []
        val_losses = []

        # Train step
        for images, _ in tqdm.tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            loss = objective.train_step(images)
            train_losses.append(loss)

        # Val step
        for images, _ in tqdm.tqdm(test_loader, desc=f"Test {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            loss = objective.val_step(images)
            val_losses.append(loss)

        # TODO add the values by fusing the eval and train metrics
        metrics = {}

        wandb.log(
            metrics,
            step=epoch,
        )

        # should_log = epoch % log_sample_every == log_sample_every - 1 or epoch == 0
        # if should_log:
        #    log_reconstructions(model, sample_images, epoch)
        #    log_generations(model, device, epoch)

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

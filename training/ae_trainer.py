"""Autoencoder training loop."""

from collections.abc import Sized
from pathlib import Path

import torch
import tqdm
from rtpt import RTPT

from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective
from utils.checkpoints import intermediate_checkpoint_path, train_state_path
from utils.config import AERunConfig
from utils.wandb_utils import log_checkpoint_artifact, log_images, log_scalar_metrics


def train_autoencoder(
    objective: AbstractObjective,
    device: torch.device,
    cfg: AERunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
    resume: bool = False,
) -> None:
    epochs = cfg.training.epochs
    log_sample_every = 10
    checkpoint_every = 25

    intermediate_ckpt_path = intermediate_checkpoint_path(
        cfg.model.model_type, cfg.dataset.name
    )
    intermediate_trainstate_path = train_state_path(intermediate_ckpt_path)

    start_epoch = 0
    if resume:
        start_epoch = objective.load_train_state(
            intermediate_trainstate_path, device=device
        )
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(
                f"--resume given but no training state found at "
                f"{intermediate_trainstate_path}; starting from epoch 0"
            )

    dataset = test_loader.dataset

    if not isinstance(dataset, Sized):
        raise TypeError("Test dataset must be sized")

    dataset_len = len(dataset)

    sample_indices = torch.randperm(dataset_len)[: min(10, dataset_len)]
    sample_images = torch.stack(
        [test_loader.dataset[i.item()][0] for i in sample_indices]
    ).to(device)

    log_images("samples/input", sample_images, step=0)

    train_metrics = MetricsCollector()
    val_metrics = MetricsCollector()

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, epochs):
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

        log_scalar_metrics(avg_train_loss, avg_val_loss, step=epoch)

        train_metrics.reset()
        val_metrics.reset()

        if epoch % log_sample_every == 0:
            reconstructed_samples = objective.sample(sample_images)
            log_images("samples/reconstructed", reconstructed_samples, step=epoch)

        if epoch % checkpoint_every == 0 and epoch > 0:
            # TODO maybe refactor this into a function??
            objective.save_checkpoint(intermediate_ckpt_path)
            objective.save_train_state(intermediate_trainstate_path, epoch)

            log_checkpoint_artifact(
                intermediate_ckpt_path,
                name=intermediate_ckpt_path.stem,
                type="autoencoder",
                description=f"Epoch: {epoch + 1}",
            )

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    # Final reconstructions
    final_reconstructed_samples = objective.sample(sample_images)
    log_images("samples/reconstructed", final_reconstructed_samples, step=epoch)

    name = f"{cfg.model.model_type}_{cfg.dataset.name}"
    ckpt_path = Path("checkpoints") / f"{name}.pt"
    objective.save_checkpoint(ckpt_path)

    log_checkpoint_artifact(ckpt_path, name=name, type="autoencoder")

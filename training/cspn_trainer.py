"""CSPN training loop."""

from pathlib import Path

import torch
import tqdm
from rtpt import RTPT

import wandb
from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective
from utils.config import CSPNRunConfig, CSPNEncoderType


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
    checkpoint_every = 25

    sample_count = max(1, min(16, cfg.dataset.num_classes))

    if cfg.model.encoder_config.encoder_type == CSPNEncoderType.CATEGORICAL:
        sample_labels = (
            torch.arange(sample_count)
            .repeat_interleave(3)
            .to(device, non_blocking=True)
        )
    elif cfg.model.encoder_config.encoder_type == CSPNEncoderType.MULTI_BINARY:
        base_vector = torch.zeros(40)
        glasses_idx = 15
        male_idx = 20
        bald_idx = 4

        glasses_vector = base_vector.clone()
        glasses_vector[glasses_idx] = 1.0
        male_vector = base_vector.clone()
        male_vector[male_idx] = 1.0
        bald_vector = base_vector.clone()
        bald_vector[bald_idx] = 1.0

        sample_labels = torch.stack(
            [base_vector, glasses_vector, male_vector, bald_vector]
        ).to(device, non_blocking=True)

    else:
        # TODO update the hardcoded colourmnist values
        sample_labels = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [8, 0, 0],
                [9, 0, 0],
            ]
        ).to(device, non_blocking=True)

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

        if epoch % checkpoint_every == 0 and epoch > 0:
            # TODO maybe this can be refactored into a function
            intermediate_name = (
                f"intermediate_{cfg.model.model_type}_{cfg.dataset.name}"
            )
            intermediate_ckpt_path = (
                Path("checkpoints/intermediate") / f"{intermediate_name}.ckpt"
            )
            objective.save_checkpoint(intermediate_ckpt_path)
            print(f"Saved intermediate checkpoint: {intermediate_ckpt_path}")

            intermediate_artifact = wandb.Artifact(
                name=intermediate_name,
                type="cspn",
                description=f"Epoch {epoch + 1}",
            )
            intermediate_artifact.add_file(str(intermediate_ckpt_path))
            wandb.log_artifact(intermediate_artifact)

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    # Final samples
    final_samples = objective.sample(sample_labels)
    final_samples_u8 = (final_samples.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {
            "samples/cspn_generated_images": [
                wandb.Image(img) for img in final_samples_u8
            ]
        },
        step=epoch,
    )

    name = f"final_{cfg.model.model_type}_{cfg.dataset.name}"
    ckpt_path = Path("checkpoints") / f"{name}.ckpt"
    objective.save_checkpoint(ckpt_path)

    artifact = wandb.Artifact(name=name, type="cspn")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)

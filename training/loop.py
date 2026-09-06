"""Generic training loop shared by autoencoder, CSPN, and LabelPC training."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
from rtpt import RTPT

from training.early_stopping import EarlyStopping
from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective, Batch
from utils.checkpoints import train_state_path
from utils.wandb_utils import log_checkpoint_artifact, log_images, log_scalar_metrics


@dataclass
class CheckpointSpec:
    intermediate_path: Path
    final_path: Path
    artifact_type: str


def run_training_loop(
    objective: AbstractObjective,
    device: torch.device,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
    checkpoint: CheckpointSpec,
    resume: bool = False,
    sample_probe: torch.Tensor | None = None,
    sample_log_key: str = "samples/generated",
    log_sample_every: int = 10,
    checkpoint_every: int = 25,
    needs_images: bool = True,
    early_stopping: EarlyStopping | None = None,
    early_stopping_metric: str = "total",
) -> None:
    """`early_stopping` also turns the final checkpoint into the *best* one rather than
    the last: the weights it kept are restored before the final save, so stopping early
    never costs you the epochs that came before the plateau."""
    intermediate_trainstate_path = train_state_path(checkpoint.intermediate_path)

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

    train_metrics = MetricsCollector()
    val_metrics = MetricsCollector()

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, epochs):
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Train {epoch + 1}/{epochs}"
        ):
            batch = Batch(
                images=images if needs_images else None, labels=labels
            ).to(device)
            loss = objective.train_step(batch)
            train_metrics.update(loss)

        avg_train_loss = train_metrics.compute_average_metrics()
        print(f"Train Loss: {avg_train_loss}")

        for images, labels in tqdm.tqdm(test_loader, desc=f"Test {epoch + 1}/{epochs}"):
            batch = Batch(
                images=images if needs_images else None, labels=labels
            ).to(device)
            loss = objective.val_step(batch)
            val_metrics.update(loss)

        avg_val_loss = val_metrics.compute_average_metrics()
        print(f"Val Loss: {avg_val_loss}")

        log_scalar_metrics(avg_train_loss, avg_val_loss, step=epoch)
        train_metrics.reset()
        val_metrics.reset()

        stop = False
        if early_stopping is not None:
            if early_stopping_metric not in avg_val_loss:
                raise KeyError(
                    f"early stopping watches {early_stopping_metric!r}, which this "
                    f"objective does not report; it has {sorted(avg_val_loss)}"
                )
            stop = early_stopping.step(
                float(avg_val_loss[early_stopping_metric]), objective.model, epoch
            )

        if sample_probe is not None and epoch % log_sample_every == 0:
            samples = objective.sample(sample_probe)
            log_images(sample_log_key, samples, step=epoch)

        if epoch % checkpoint_every == 0 and epoch > 0:
            objective.save_checkpoint(checkpoint.intermediate_path)
            objective.save_train_state(intermediate_trainstate_path, epoch)
            print(f"Saved intermediate checkpoint: {checkpoint.intermediate_path}")

            log_checkpoint_artifact(
                checkpoint.intermediate_path,
                name=checkpoint.intermediate_path.stem,
                type=checkpoint.artifact_type,
                description=f"Epoch {epoch + 1}",
            )

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

        if stop:
            assert early_stopping is not None
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement in "
                f"{early_stopping_metric} for {early_stopping.patience} epochs "
                f"(best {early_stopping.best_loss:.4f} at epoch "
                f"{early_stopping.best_epoch + 1})"
            )
            break

    if early_stopping is not None and early_stopping.restore_best_weights(
        objective.model
    ):
        print(
            f"Restored weights from epoch {early_stopping.best_epoch + 1} "
            f"({early_stopping_metric} {early_stopping.best_loss:.4f})"
        )

    if sample_probe is not None:
        final_samples = objective.sample(sample_probe)
        log_images(sample_log_key, final_samples, step=epoch)

    objective.save_checkpoint(checkpoint.final_path)
    log_checkpoint_artifact(
        checkpoint.final_path,
        name=checkpoint.final_path.stem,
        type=checkpoint.artifact_type,
    )

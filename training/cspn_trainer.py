"""CSPN training loop."""

from pathlib import Path

import torch
import tqdm
from rtpt import RTPT

from models.cspn.psinet.label_pc import LabelPC
from training.metrics import MetricsCollector
from training.objectives.base import AbstractObjective
from utils.checkpoints import intermediate_checkpoint_path, train_state_path
from utils.config import CSPNEncoderType, CSPNRunConfig
from utils.wandb_utils import log_checkpoint_artifact, log_images, log_scalar_metrics


def _themed_multi_binary_labels(
    themes: list[dict[int, float]],
    num_attributes: int,
    device: torch.device,
    label_pc: LabelPC | None,
) -> torch.Tensor:
    """Builds one label vector per theme (a sparse {attribute_idx: value} spec)."""
    if label_pc is not None:
        return torch.cat(
            [
                label_pc.complete_partial(known, batch_size=1, device=device)
                for known in themes
            ],
            dim=0,
        )

    print(
        "No LabelPC available -- falling back to zero-filled (attribute=off) labels "
        "for sample logging."
    )
    vectors = []
    for known in themes:
        vector = torch.zeros(num_attributes)
        for idx, value in known.items():
            vector[idx] = value
        vectors.append(vector)
    return torch.stack(vectors).to(device, non_blocking=True)


def train_cspn(
    objective: AbstractObjective,
    device: torch.device,
    cfg: CSPNRunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
    resume: bool = False,
    label_pc: LabelPC | None = None,
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

    sample_count = max(1, min(16, cfg.dataset.num_classes))

    if cfg.model.encoder_config.encoder_type == CSPNEncoderType.CATEGORICAL:
        sample_labels = (
            torch.arange(sample_count)
            .repeat_interleave(3)
            .to(device, non_blocking=True)
        )
    elif cfg.model.encoder_config.encoder_type == CSPNEncoderType.MULTI_BINARY:
        glasses_idx = 15
        male_idx = 20
        bald_idx = 4

        # {} = fully unconditional (all attributes marginalized/sampled by LabelPC,
        # or zero-filled in the no-LabelPC fallback).
        themes: list[dict[int, float]] = [
            {},
            {glasses_idx: 1.0},
            {male_idx: 1.0},
            {bald_idx: 1.0},
        ]
        sample_labels = _themed_multi_binary_labels(
            themes, cfg.dataset.num_classes, device, label_pc
        )

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

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, epochs):
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

        log_scalar_metrics(avg_train_loss, avg_val_loss, step=epoch)
        train_metrics.reset()
        val_metrics.reset()

        if epoch % log_sample_every == 0:
            samples = objective.sample(sample_labels)
            log_images("samples/cspn_generated_images", samples, step=epoch)

        if epoch % checkpoint_every == 0 and epoch > 0:
            # TODO maybe this can be refactored into a function
            objective.save_checkpoint(intermediate_ckpt_path)
            objective.save_train_state(intermediate_trainstate_path, epoch)
            print(f"Saved intermediate checkpoint: {intermediate_ckpt_path}")

            log_checkpoint_artifact(
                intermediate_ckpt_path,
                name=intermediate_ckpt_path.stem,
                type="cspn",
                description=f"Epoch {epoch + 1}",
            )

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    # Final samples
    final_samples = objective.sample(sample_labels)
    log_images("samples/cspn_generated_images", final_samples, step=epoch)

    name = f"{cfg.model.model_type}_{cfg.dataset.name}"
    ckpt_path = Path("checkpoints") / f"{name}.pt"
    objective.save_checkpoint(ckpt_path)

    log_checkpoint_artifact(ckpt_path, name=name, type="cspn")

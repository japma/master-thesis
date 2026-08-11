"""LabelPC training loop."""

from __future__ import annotations

import torch
import tqdm
from rtpt import RTPT

import wandb
from training.metrics import MetricsCollector
from training.objectives.label_pc_objective import LabelPCObjective
from utils.config import CSPNRunConfig


def train_label_pc(
    objective: LabelPCObjective,
    device: torch.device,
    cfg: CSPNRunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    rtpt: RTPT,
) -> None:
    epochs = cfg.training.epochs

    train_metrics = MetricsCollector()
    val_metrics = MetricsCollector()

    for epoch in range(epochs):
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Train {epoch + 1}/{epochs}"
        ):
            labels = labels.to(device, non_blocking=True)
            loss = objective.train_step(images, labels)
            train_metrics.update(loss)

        avg_train_loss = train_metrics.compute_average_metrics()
        print(f"Train Loss: {avg_train_loss}")

        for images, labels in tqdm.tqdm(test_loader, desc=f"Test {epoch + 1}/{epochs}"):
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

        objective.on_epoch_end()
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

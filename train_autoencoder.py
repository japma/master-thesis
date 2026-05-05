"""Hydra entrypoint for autoencoder training."""

import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from hydra import main
from rtpt import RTPT
from tqdm import tqdm

from models.autoencoder.variational_autoencoder import VariationalAutoencoder
from utils import (
    create_run_directories,
    format_elapsed_time,
    get_data_loaders,
    save_checkpoint,
    seed_everything,
)
from utils.config import parse_autoencoder_train_config
from utils.train import resolve_device

logger = logging.getLogger(__name__)


def _build_autoencoder(cfg, device):
    model_cfg = cfg.model
    if model_cfg.get("autoencoder") is not None:
        model_cfg = model_cfg.autoencoder

    return VariationalAutoencoder(
        input_size=cfg.data.input_size,
        latent_size=cfg.data.latent_size,
        image_shape=cfg.data.image_shape,
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        reconstructed = model(images)
        target = images.view(images.size(0), -1) if reconstructed.dim() == 2 else images
        loss = criterion(reconstructed, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            reconstructed = model(images)
            target = images.view(images.size(0), -1) if reconstructed.dim() == 2 else images
            loss = criterion(reconstructed, target)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def run_autoencoder_training(cfg):
    start_time = time.perf_counter()

    dataset_name = cfg.data.name
    output_dir = cfg.run_dir

    device = resolve_device()
    logger.info(
        "Device: %s, Dataset: %s, Output Dir: %s", device, dataset_name, output_dir
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_name}_autoencoder_training",
        max_iterations=max(cfg.training.epochs, 1),
    )
    rtpt.start()

    run_dirs = create_run_directories(output_dir)
    logger.info("Run directory: %s", run_dirs.run_dir)

    train_loader, test_loader = get_data_loaders(
        dataset_name, cfg.training.batch_size, dataset_kwargs=cfg.data.dataset_kwargs
    )

    ae_epochs = cfg.training.epochs
    ae_learning_rate = cfg.training.learning_rate

    model = _build_autoencoder(cfg, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ae_learning_rate)

    logger.info("Training autoencoder for %s epochs...", ae_epochs)
    for epoch in range(ae_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        logger.info(
            "AE Epoch %s/%s - Train Loss: %.4f, Test Loss: %.4f",
            epoch + 1,
            ae_epochs,
            train_loss,
            test_loss,
        )

        rtpt.step(subtitle=f"AE {epoch + 1}/{ae_epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "autoencoder")

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    elapsed_formatted = format_elapsed_time(elapsed_seconds)
    logger.info(
        "Autoencoder training completed in %s (%.2fs)",
        elapsed_formatted,
        elapsed_seconds,
    )


@main(version_base=None, config_path="configs", config_name="train_autoencoder")
def main_hydra(cfg) -> None:
    seed = seed_everything(cfg.get("seed"))
    logger.info("Using seed: %s", seed)
    run_autoencoder_training(parse_autoencoder_train_config(cfg))


if __name__ == "__main__":
    main_hydra()

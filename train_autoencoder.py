"""Autoencoder training."""

import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from rtpt import RTPT
from tqdm import tqdm

from models import VariationalAutoencoder
from utils import (
    create_run_directories,
    save_checkpoint,
)
from dataset_loaders import get_data_loaders
from utils.tracking import WandbTracker
from utils.train import resolve_device


def _build_autoencoder(cfg, device):
    ae_cfg = cfg.autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    return VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        base_channels=ae_cfg.base_channels,
        num_blocks=ae_cfg.num_blocks,
        res_blocks=ae_cfg.res_blocks,
    ).to(device)


def _vae_loss(images, recon, mu, logvar, beta=1.0):
    """ELBO loss: reconstruction (MSE) + β · KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, images, reduction="sum") / images.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = _vae_loss(images, recon, mu, logvar, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    n = len(train_loader)
    return total_loss / n, total_recon / n, total_kl / n


def evaluate(model, test_loader, device, beta=1.0):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = _vae_loss(images, recon, mu, logvar, beta=beta)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    n = len(test_loader)
    return total_loss / n, total_recon / n, total_kl / n


def run_autoencoder_training(cfg):
    dataset_cfg = cfg.dataset
    hydra_cfg = HydraConfig().get()
    output_dir = hydra_cfg.runtime.output_dir

    device = resolve_device()

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_cfg.name}_autoencoder_training",
        max_iterations=max(cfg.training.epochs, 1),
    )
    rtpt.start()

    wandb_run = WandbTracker(cfg)

    run_dirs = create_run_directories(output_dir)

    train_loader, test_loader = get_data_loaders(
        dataset_cfg,
        cfg.training.batch_size,
    )

    ae_epochs = cfg.training.epochs
    ae_learning_rate = cfg.training.learning_rate
    warmup_epochs = max(1, ae_epochs // 4)

    model = _build_autoencoder(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=ae_learning_rate)

    for epoch in range(ae_epochs):
        beta = min(1.0, (epoch + 1) / warmup_epochs)

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta=beta
        )
        test_loss, test_recon, test_kl = evaluate(model, test_loader, device, beta=beta)

        wandb_run.log(
            {
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "test_loss": test_loss,
                "test_recon": test_recon,
                "test_kl": test_kl,
                "beta": beta,
            }
        )
        rtpt.step(subtitle=f"AE {epoch + 1}/{ae_epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "autoencoder")

    wandb_run.finish()

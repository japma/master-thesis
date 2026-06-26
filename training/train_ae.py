"""Autoencoder training loop."""

import torch
import tqdm
from torch import nn

import wandb
from models.autoencoder import AbstractAutoencoder
from training.losses import kl_per_dim
from utils.config import AERunConfig


def _train_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    warmup_epochs: int = 10,
) -> tuple[float, float, float]:
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)

    annealed_beta = min(1.0, epoch / max(warmup_epochs, 1))

    for images, _ in tqdm.tqdm(loader, desc=f"Train {epoch + 1}/{epochs}"):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()

        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = loss_fn(
            images=images, recon=recon, mu=mu, logvar=logvar, beta=annealed_beta
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.detach()
        total_recon += recon_loss.detach()
        total_kl += kl_loss.detach()

    n = len(loader)
    return (total_loss / n).item(), (total_recon / n).item(), (total_kl / n).item()


def _val_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float, float, torch.Tensor]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_kl_per_dim = torch.zeros(
        model.get_latent_dim(), device=device, dtype=torch.float
    )

    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc=f"Val   {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = loss_fn(
                images=images, recon=recon, mu=mu, logvar=logvar
            )
            total_loss += loss.detach()
            total_recon += recon_loss.detach()
            total_kl += kl_loss.detach()
            batch_kl_dims = kl_per_dim(mu, logvar)
            total_kl_per_dim += batch_kl_dims

    n = len(loader)
    return (
        (total_loss / n).item(),
        (total_recon / n).item(),
        (total_kl / n).item(),
        (total_kl_per_dim / n),
    )


def _log_reconstructions(
    model: AbstractAutoencoder,
    sample_images: torch.Tensor,
    epoch: int,
) -> None:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(sample_images)
        recon = torch.sigmoid(logits)
    recon_u8 = (recon.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/recon_images": [wandb.Image(img) for img in recon_u8]},
        step=epoch,
    )


def train_autoencoder(
    model: AbstractAutoencoder,
    device: torch.device,
    cfg: AERunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    rtpt,
) -> None:
    model.to(device)

    epochs = cfg.training.epochs
    log_sample_every = 10

    sample_images = next(iter(test_loader))[0][:16].to(device)
    sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/input": [wandb.Image(img) for img in sample_images_u8]},
        step=0,
    )

    warmup_epochs = 10

    for epoch in range(epochs):
        train_loss, train_recon, train_kl = _train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            epochs,
            warmup_epochs=warmup_epochs,
        )

        val_loss, val_recon, val_kl, kl_dims = _val_epoch(
            model, test_loader, loss_fn, device, epoch, epochs
        )

        scheduler.step()

        active_dims = (kl_dims > 0.1).sum().item()

        annealed_beta = min(1.0, epoch / max(warmup_epochs, 1))

        wandb.log(
            {
                "train_loss": train_loss,
                "train_recon_loss": train_recon,
                "train_kl_loss": train_kl,
                "val_loss": val_loss,
                "val_recon_loss": val_recon,
                "val_kl_loss": val_kl,
                "val_kl_per_dim": wandb.Histogram(kl_dims.cpu().tolist()),
                "active_dims": active_dims,
                "kl_beta": annealed_beta,
            },
            step=epoch,
        )

        if epoch % log_sample_every == log_sample_every - 1 or epoch == 0:
            _log_reconstructions(model, sample_images, epoch)

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

"""Autoencoder training loop."""

from training.losses import vae_loss
import torch
import tqdm
import wandb
from torch import nn

from models.autoencoder import AbstractAutoencoder
from training.losses import beta_vae_loss


def _beta_for_epoch(
    epoch: int,
    beta_start: float,
    beta_end: float,
    anneal_epochs: int,
) -> float:
    if anneal_epochs <= 1:
        return beta_end
    progress = min(epoch / (anneal_epochs - 1), 1.0)
    return beta_start + progress * (beta_end - beta_start)


def _train_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    beta: float,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)

    for images, _ in tqdm.tqdm(loader, desc=f"Train {epoch + 1}/{epochs}"):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()

        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = beta_vae_loss(
            images, recon, mu, logvar, recon_loss_fn=loss_fn, beta=beta
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
    beta: float,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc=f"Val   {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            recon, mu, logvar = model(images)
            # loss, recon_loss, kl_loss = beta_vae_loss(
            loss, recon_loss, kl_loss = vae_loss(
                images, recon, mu, logvar, recon_loss_fn=loss_fn, beta=beta
            )
            total_loss += loss.detach()
            total_recon += recon_loss.detach()
            total_kl += kl_loss.detach()

    n = len(loader)
    return (total_loss / n).item(), (total_recon / n).item(), (total_kl / n).item()


def _log_reconstructions(
    model: AbstractAutoencoder,
    sample_images: torch.Tensor,
    epoch: int,
) -> None:
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(sample_images)
    recon_u8 = (recon.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/recon_images": [wandb.Image(img) for img in recon_u8]},
        step=epoch,
    )


def train_autoencoder(
    model: AbstractAutoencoder,
    device: torch.device,
    cfg,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    rtpt,
) -> None:
    model.to(device)

    epochs = cfg.training.epochs
    beta_start = cfg.training.beta_start
    beta_end = cfg.training.beta_end
    beta_anneal_epochs = min(cfg.training.beta_anneal_epochs, epochs)
    log_sample_every = 10

    sample_images = next(iter(train_loader))[0][:16].to(device)
    sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/input": [wandb.Image(img) for img in sample_images_u8]},
        step=0,
    )

    for epoch in range(epochs):
        beta = _beta_for_epoch(epoch, beta_start, beta_end, beta_anneal_epochs)

        train_loss, train_recon, train_kl = _train_epoch(
            model, train_loader, optimizer, loss_fn, beta, device, epoch, epochs
        )

        val_loss, val_recon, val_kl = _val_epoch(
            model, test_loader, loss_fn, beta, device, epoch, epochs
        )

        scheduler.step()

        wandb.log(
            {
                "train_loss": train_loss,
                "train_recon_loss": train_recon,
                "train_kl_loss": train_kl,
                "val_loss": val_loss,
                "val_recon_loss": val_recon,
                "val_kl_loss": val_kl,
                "beta": beta,
            },
            step=epoch,
        )

        if epoch % log_sample_every == log_sample_every - 1 or epoch == 0:
            _log_reconstructions(model, sample_images, epoch)

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

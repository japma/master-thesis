"""Autoencoder training loop."""

import torch
import tqdm

import wandb
from models.autoencoder import AbstractAutoencoder
from training.losses import VAELoss, kl_per_dim
from utils.config import AERunConfig


def _train_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: VAELoss,
    device: torch.device,
    epoch: int,
    epochs: int,
    annealed_beta: float,
) -> tuple[float, float, float, float]:
    """Run one training epoch.

    Returns:
        mean_total, mean_recon, mean_kl, mean_perceptual
    """
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_perc = torch.tensor(0.0, device=device)

    for images, _ in tqdm.tqdm(loader, desc=f"Train {epoch + 1}/{epochs}"):
        images = images.to(device, non_blocking=True)

        logits, mu, logvar = model(images)
        out = loss_fn(
            images=images,
            logits=logits,
            mu=mu,
            logvar=logvar,
            beta=annealed_beta,
        )

        optimizer.zero_grad()
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += out["total"].detach()
        total_recon += out["recon"].detach()
        total_kl += out["kl"].detach()
        total_perc += out["perceptual"].detach()

    n = len(loader)
    return (
        (total_loss / n).item(),
        (total_recon / n).item(),
        (total_kl / n).item(),
        (total_perc / n).item(),
    )


def _val_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    loss_fn: VAELoss,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float, float, torch.Tensor]:
    """Run one validation epoch using only recon + KL (no perceptual)."""
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)

    total_samples = 0
    total_kl_per_dim = torch.zeros(
        model.get_latent_dim(), device=device, dtype=torch.float
    )

    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc=f"Val   {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            B = images.size(0)

            logits, mu, logvar = model(images)
            total, recon_loss, kl_loss = loss_fn.beta_vae(
                images=images, recon=logits, mu=mu, logvar=logvar
            )

            total_loss += total.detach()
            total_recon += recon_loss.detach()
            total_kl += kl_loss.detach()

            batch_kl_dims = kl_per_dim(mu, logvar)
            total_kl_per_dim += batch_kl_dims * B
            total_samples += B

    n = len(loader)
    return (
        (total_loss / n).item(),
        (total_recon / n).item(),
        (total_kl / n).item(),
        total_kl_per_dim / total_samples,
    )


def _log_reconstructions(
    model: AbstractAutoencoder,
    sample_images: torch.Tensor,
    epoch: int,
) -> None:
    """Log reconstructions of fixed validation images."""
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(sample_images)
        recon = torch.sigmoid(logits)
    recon_u8 = (recon.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/recon_images": [wandb.Image(img) for img in recon_u8]},
        step=epoch,
    )


def _log_generations(
    model: AbstractAutoencoder,
    device: torch.device,
    epoch: int,
    n: int = 16,
) -> None:
    """Decode random latent vectors as a latent space health diagnostic."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.get_latent_dim(), device=device)
        gen = model.decode(z)
    gen_u8 = (gen.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/generated": [wandb.Image(img) for img in gen_u8]},
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
    loss_fn: VAELoss,
    rtpt,
) -> None:
    model.to(device)
    loss_fn.to(device)

    epochs = cfg.training.epochs
    warmup_epochs = cfg.training.kl_warmup_epochs
    log_sample_every = 10

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

        train_loss, train_recon, train_kl, train_perc = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            epochs=epochs,
            annealed_beta=annealed_beta,
        )

        val_loss, val_recon, val_kl, kl_dims = _val_epoch(
            model, test_loader, loss_fn, device, epoch, epochs
        )

        scheduler.step()

        active_dims = (kl_dims > 0.1).sum().item()

        wandb.log(
            {
                "train/loss": train_loss,
                "train/recon": train_recon,
                "train/kl": train_kl,
                "train/perceptual": train_perc,
                "val/loss": val_loss,
                "val/recon": val_recon,
                "val/kl": val_kl,
                # "val/kl_per_dim":  wandb.Histogram(kl_dims.cpu().tolist()),
                "latent/active_dims": active_dims,
                "latent/kl_beta": annealed_beta,
            },
            step=epoch,
        )

        should_log = epoch % log_sample_every == log_sample_every - 1 or epoch == 0
        if should_log:
            _log_reconstructions(model, sample_images, epoch)
            _log_generations(model, device, epoch)

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

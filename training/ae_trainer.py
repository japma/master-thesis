"""Autoencoder training loop."""

import torch
import tqdm
from torch import nn

import wandb
from models.autoencoder import AbstractAutoencoder
from training.losses import VAELoss, kl_per_dim
from utils.config import AERunConfig


def _train_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_fn: VAELoss,
    device: torch.device,
    epoch: int,
    epochs: int,
    annealed_beta: float,
    global_step: int,
) -> tuple[float, float, float, float, float, int]:
    """Run one training epoch.

    Returns:
        mean_total_g, mean_recon, mean_kl, mean_perceptual, mean_adv_g,
        updated global_step
    """
    model.train()

    total_g = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_perc = torch.tensor(0.0, device=device)
    total_adv_g = torch.tensor(0.0, device=device)

    for images, _ in tqdm.tqdm(loader, desc=f"Train {epoch + 1}/{epochs}"):
        images = images.to(device, non_blocking=True)

        # --- Generator step (VAE) ---
        logits, mu, logvar = model(images)
        out = loss_fn.generator_loss(
            images=images,
            logits=logits,
            mu=mu,
            logvar=logvar,
            last_decoder_layer=model.output_proj,
            step=global_step,
            beta=annealed_beta,
        )

        optimizer_g.zero_grad()
        out["total_g"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_g.step()

        # --- Discriminator step ---
        recon_img = torch.sigmoid(logits).detach()
        d_loss = loss_fn.discriminator_loss(images, recon_img)

        optimizer_d.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_fn.discriminator.parameters(), max_norm=1.0)
        optimizer_d.step()

        total_g += out["total_g"].detach()
        total_recon += out["recon"].detach()
        total_kl += out["kl"].detach()
        total_perc += out["perceptual"].detach()
        total_adv_g += out["adversarial_g"].detach()

        global_step += 1

    n = len(loader)
    return (
        (total_g / n).item(),
        (total_recon / n).item(),
        (total_kl / n).item(),
        (total_perc / n).item(),
        (total_adv_g / n).item(),
        global_step,
    )


def _val_epoch(
    model: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    loss_fn: VAELoss,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float, float, torch.Tensor]:
    """Run one validation epoch.

    Uses only the beta_vae component (recon + KL) — no perceptual or
    adversarial — so that:
      - val loss is comparable across all epochs regardless of warmup state
      - autograd.grad (used by adaptive weighting) is never called under
        no_grad, which would crash
    """
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

            batch_kl_dims = kl_per_dim(mu, logvar)  # (latent_dim,)
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
    """Log side-by-side reconstructions of fixed validation images."""
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
    """Sample random latent vectors and decode them."""
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
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.LRScheduler,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
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

    global_step = 0

    for epoch in range(epochs):
        annealed_beta = min(1.0, epoch / max(warmup_epochs, 1))

        (
            train_g,
            train_recon,
            train_kl,
            train_perc,
            train_adv_g,
            global_step,
        ) = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            epochs=epochs,
            annealed_beta=annealed_beta,
            global_step=global_step,
        )

        val_loss, val_recon, val_kl, kl_dims = _val_epoch(
            model, test_loader, loss_fn, device, epoch, epochs
        )

        scheduler_g.step()
        scheduler_d.step()

        active_dims = (kl_dims > 0.1).sum().item()

        wandb.log(
            {
                "train/loss_g": train_g,
                "train/recon": train_recon,
                "train/kl": train_kl,
                "train/perceptual": train_perc,
                "train/adv_g": train_adv_g,
                "val/loss": val_loss,
                "val/recon": val_recon,
                "val/kl": val_kl,
                # "val/kl_per_dim":  wandb.Histogram(kl_dims.cpu().tolist()),
                "latent/active_dims": active_dims,
                "latent/kl_beta": annealed_beta,
                "train/global_step": global_step,
            },
            step=epoch,
        )

        should_log = epoch % log_sample_every == log_sample_every - 1 or epoch == 0
        if should_log:
            _log_reconstructions(model, sample_images, epoch)
            _log_generations(model, device, epoch)

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

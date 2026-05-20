import torch
import tqdm
import torch.nn.functional as F
from rtpt import RTPT
from torch import nn
from typing import Any

import wandb
from losses import vae_loss, negative_log_likelihood_loss
from models.autoencoder import AbstractAutoencoder
from models.cspn import AbstractCSPN


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


def _build_cspn_context(model: nn.Module, labels: torch.Tensor) -> torch.Tensor:
    cond_net: Any = getattr(model, "cond_net", None)
    if cond_net is not None:
        mlp: Any = getattr(cond_net, "mlp", None)
        context_dim = mlp[0].in_features
        return F.one_hot(labels, num_classes=context_dim).float()

    return labels.view(-1, 1).float()


def train_autoencoder(
    model: AbstractAutoencoder,
    device: torch.device,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    beta_start: float,
    beta_end: float,
    beta_anneal_epochs: int,
    rtpt: RTPT,
) -> None:
    model.to(device)

    # Enable CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    sample_images = next(iter(train_loader))[0][:16].to(device)
    sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {
            "samples/input": [wandb.Image(sample) for sample in sample_images_u8],
        },
        step=0,
    )
    for epoch in range(epochs):
        beta = _beta_for_epoch(
            epoch=epoch,
            beta_start=beta_start,
            beta_end=beta_end,
            anneal_epochs=min(beta_anneal_epochs, epochs),
        )
        model.train()
        total_train_loss = torch.tensor(0.0, device=device)
        total_train_recon = torch.tensor(0.0, device=device)
        total_train_kl = torch.tensor(0.0, device=device)
        for images, _ in tqdm.tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(
                images, recon, mu, logvar, recon_loss_fn=loss_fn, beta=beta
            )
            loss.backward()
            optimizer.step()
            # Accumulate without .item() to avoid GPU-CPU sync every batch
            total_train_loss += loss.detach()
            total_train_recon += recon_loss.detach()
            total_train_kl += kl_loss.detach()

        n_train = len(train_loader)
        avg_train_loss = (total_train_loss / n_train).item()
        avg_train_recon = (total_train_recon / n_train).item()
        avg_train_kl = (total_train_kl / n_train).item()

        model.eval()
        total_val_loss = torch.tensor(0.0, device=device)
        total_val_recon = torch.tensor(0.0, device=device)
        total_val_kl = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for images, _ in tqdm.tqdm(
                test_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device, non_blocking=True)
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(
                    images, recon, mu, logvar, recon_loss_fn=loss_fn, beta=beta
                )

                # Accumulate without .item()
                total_val_loss += loss.detach()
                total_val_recon += recon_loss.detach()
                total_val_kl += kl_loss.detach()

            if epoch % 10 == 9 or epoch == 0:
                recon_images, _, _ = model(sample_images)
                recon_images_u8 = (recon_images.clamp(0, 1) * 255).byte().cpu()

                wandb.log(
                    {
                        "samples/recon_images": [
                            wandb.Image(recon) for recon in recon_images_u8
                        ],
                    },
                    step=epoch,
                )

            n_val = len(test_loader)
            avg_val_loss = (total_val_loss / n_val).item()
            avg_val_recon = (total_val_recon / n_val).item()
            avg_val_kl = (total_val_kl / n_val).item()

            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "train_recon_loss": avg_train_recon,
                    "train_kl_loss": avg_train_kl,
                    "val_loss": avg_val_loss,
                    "val_recon_loss": avg_val_recon,
                    "val_kl_loss": avg_val_kl,
                    "beta": beta,
                },
                step=epoch,
            )
        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")


def train_cspn(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    device: torch.device,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    rtpt: RTPT,
):
    model.to(device)
    autoencoder.to(device)
    autoencoder.eval()

    # Enable CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for epoch in range(epochs):
        model.train()
        total_train_loss = torch.tensor(0.0, device=device)
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            context = _build_cspn_context(model, labels).to(device, non_blocking=True)
            with torch.no_grad():
                latent = autoencoder.encode(images)

            outputs = model(latent, context)
            loss = negative_log_likelihood_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate without .item()
            total_train_loss += loss.detach()

        n_train = len(train_loader)
        avg_train_loss = (total_train_loss / n_train).item()

        model.eval()
        total_val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for images, labels in tqdm.tqdm(
                test_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                latent = autoencoder.encode(images)
                context = _build_cspn_context(model, labels).to(device, non_blocking=True)
                outputs = model(latent, context)
                loss = negative_log_likelihood_loss(outputs)
                # Accumulate without .item()
                total_val_loss += loss.detach()

        if epoch % 10 == 9 or epoch == 0:
            num_classes = 10
            samples_per_class = 3
            sample_labels = torch.arange(num_classes, device=device).repeat(
                samples_per_class
            )
            sample_context = _build_cspn_context(model, sample_labels).to(device)

            with torch.no_grad():
                sampled_latent = model.sample(sample_context)
                sampled_images = autoencoder.decode(sampled_latent)

            sampled_images_u8 = (sampled_images.clamp(0, 1) * 255).byte().cpu()
            wandb.log(
                {
                    "samples/cspn_generated_images": [
                        wandb.Image(sample) for sample in sampled_images_u8
                    ],
                },
                step=epoch,
            )

        n_val = len(test_loader)
        avg_val_loss = (total_val_loss / n_val).item()

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            step=epoch,
        )

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

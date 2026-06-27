"""CSPN training loop."""

import torch
import tqdm
from rtpt import RTPT

import wandb
from models.autoencoder import AbstractAutoencoder
from models.cspn import AbstractCSPN
from training.early_stopping import EarlyStopping
from training.losses import negative_log_likelihood_loss
from utils.config import CSPNRunConfig


def _train_epoch(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = torch.tensor(0.0, device=device)

    for images, labels in tqdm.tqdm(loader, desc=f"Train {epoch + 1}/{epochs}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            latent = autoencoder.encode(images)

        loss = negative_log_likelihood_loss(model(latent, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()

    return (total_loss / len(loader)).item()


def _val_epoch(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc=f"Val   {epoch + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            latent = autoencoder.encode(images)
            loss = negative_log_likelihood_loss(model(latent, labels))
            total_loss += loss.detach()

    return (total_loss / len(loader)).item()


def _log_samples(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    device: torch.device,
    num_classes: int,
    samples_per_class: int,
    epoch: int,
) -> None:
    model.eval()
    class_sample_count = min(num_classes, 16)
    sample_labels = torch.arange(class_sample_count, device=device).repeat_interleave(
        samples_per_class
    )

    with torch.no_grad():
        sampled_latent = model.sample(sample_labels)
        sampled_images = autoencoder.decode(sampled_latent)
    images_u8 = (sampled_images.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/cspn_generated_images": [wandb.Image(img) for img in images_u8]},
        step=epoch,
    )


def train_cspn(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    device: torch.device,
    cfg: CSPNRunConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rtpt: RTPT,
) -> None:
    model.to(device)
    autoencoder.to(device)
    autoencoder.eval()

    epochs = cfg.training.epochs
    log_sample_every = 10

    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta,
    )

    for epoch in range(epochs):
        train_loss = _train_epoch(
            model,
            autoencoder,
            train_loader,
            optimizer,
            device,
            epoch,
            epochs,
        )
        val_loss = _val_epoch(model, autoencoder, test_loader, device, epoch, epochs)
        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            },
            step=epoch,
        )

        if epoch % log_sample_every == log_sample_every - 1 or epoch == 0:
            _log_samples(
                model,
                autoencoder,
                device,
                num_classes=cfg.dataset.num_classes,
                samples_per_class=3,
                epoch=epoch,
            )

        if early_stopping.step(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

    early_stopping.restore_best_weights(model)

    _log_samples(
        model,
        autoencoder,
        device,
        num_classes=cfg.dataset.num_classes,
        samples_per_class=3,
        epoch=epoch,
    )
    wandb.log({"best_val_loss": early_stopping.best_loss})

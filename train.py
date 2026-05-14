import torch
import tqdm
from rtpt import RTPT

import wandb
from losses import vae_loss
from models.autoencoder import AbstractAutoencoder
from models.cspn import AbstractCSPN


def train_autoencoder(
    model: AbstractAutoencoder,
    device: torch.device,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # TODO set beta correctly
    beta: float,
    rtpt: RTPT,
) -> None:
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = total_train_recon = total_train_kl = 0.0
        for images, _ in tqdm.tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(images, recon, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            total_train_kl += kl_loss.item()

        n_train = len(train_loader)
        avg_train_loss = total_train_loss / n_train
        avg_train_recon = total_train_recon / n_train
        avg_train_kl = total_train_kl / n_train

        model.eval()
        total_val_loss = total_val_recon = total_val_kl = 0
        with torch.no_grad():
            for images, _ in tqdm.tqdm(
                test_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device)
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(
                    images, recon, mu, logvar, beta=beta
                )

                total_val_loss += loss.item()
                total_val_recon += recon_loss.item()
                total_val_kl += kl_loss.item()

            if epoch % 10 == 9 or epoch == 0:
                sample_images = images[:16]
                recon_images, _, _ = model(sample_images)
                sample_images_u8 = (sample_images.clamp(0, 1) * 255).byte().cpu()
                recon_images_u8 = (recon_images.clamp(0, 1) * 255).byte().cpu()

                wandb.log(
                    {
                        "sample_images": [
                            wandb.Image(sample) for sample in sample_images_u8
                        ],
                        "recon_images": [
                            wandb.Image(recon) for recon in recon_images_u8
                        ],
                    },
                    step=epoch,
                )

            n_val = len(test_loader)
            avg_val_loss = total_val_loss / n_val
            avg_val_recon = total_val_recon / n_val
            avg_val_kl = total_val_kl / n_val

            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "train_recon_loss": avg_train_recon,
                    "train_kl_loss": avg_train_kl,
                    "val_loss": avg_val_loss,
                    "val_recon_loss": avg_val_recon,
                    "val_kl_loss": avg_val_kl,
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
    for epoch in range(epochs):
        model.train()
        total_train_loss = total_log_prob = 0.0
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                latent = autoencoder.encode(images)

            log_prob = model(latent, labels)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_log_prob += log_prob.mean().item()

        n_train = len(train_loader)
        avg_train_loss = total_train_loss / n_train
        avg_train_log_prob = total_log_prob / n_train

        model.eval()
        total_val_loss = total_val_log_prob = 0.0
        with torch.no_grad():
            for images, _ in tqdm.tqdm(
                test_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device)
                latent = autoencoder.encode(images)
                log_prob = model(latent, images)
                loss = -log_prob.mean()

                total_val_loss += loss.item()
                total_val_loss += loss.item()
                total_val_log_prob += log_prob.mean().item()

        if epoch % 10 == 0:
            # TODO sample all classes and write to wandb
            pass

        n_val = len(test_loader)
        avg_val_loss = total_val_loss / n_val
        avg_val_log_prob = total_val_log_prob / n_val

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "train_log_prob": avg_train_log_prob,
                "val_loss": avg_val_loss,
                "val_log_prob": avg_val_log_prob,
            },
            step=epoch,
        )

        rtpt.step(subtitle=f"{epoch + 1}/{epochs}")

"""Training entrypoint for Autoencoder (VAE)."""

import torch
import tqdm
from pathlib import Path
from rtpt import RTPT
import wandb
from torch import nn

from utils.config import load_config
from models.autoencoder import create_autoencoder, AbstractAutoencoder
from dataset_loaders import build_data_loaders
from training.losses import vae_loss
from utils.utils import seed_everything, resolve_device


def _beta_for_epoch(
    epoch: int,
    beta_start: float,
    beta_end: float,
    anneal_epochs: int,
) -> float:
    """Compute beta annealing schedule for VAE training."""
    if anneal_epochs <= 1:
        return beta_end

    progress = min(epoch / (anneal_epochs - 1), 1.0)
    return beta_start + progress * (beta_end - beta_start)


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
    """Train the Variational Autoencoder model."""
    model.to(device)

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
            loss = nn.MSELoss()(recon, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.detach()
            total_train_recon += recon_loss.detach()
            total_train_kl += kl_loss.detach()

        n_train = len(train_loader)
        avg_train_loss = (total_train_loss / n_train).item()
        avg_train_recon = (total_train_recon / n_train).item()
        avg_train_kl = (total_train_kl / n_train).item()

        # Check for posterior collapse
        if avg_train_kl < 0.01:
            print(
                f"WARNING: KL loss very small ({avg_train_kl:.6f}) - possible posterior collapse!"
            )
            print("   The model may be ignoring the latent space. Try reducing beta.")
        if avg_train_recon < 0.001:
            print(
                f"WARNING: Reconstruction loss suspiciously low ({avg_train_recon:.6f}) - check if all images are identical"
            )

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


def main():
    """Train the autoencoder model."""
    cfg = load_config()

    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_cfg = cfg.dataset
    dataset_name = dataset_cfg.name
    epochs = cfg.training.epochs
    wandb_mode = cfg.wandb

    name = f"Autoencoder_{dataset_name}_seed{seed}"

    print(f"Training Autoencoder on {dataset_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)
    ae = create_autoencoder(
        model_type=cfg.autoencoder.model_type,
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        device=device,
        base_channels=cfg.autoencoder.base_channels,
        num_blocks=cfg.autoencoder.num_blocks,
        res_blocks=cfg.autoencoder.res_blocks,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=name,
        max_iterations=max(epochs, 1),
    )
    rtpt.start()

    wandb_cfg = {
        "dataset": dataset_name,
        "model": "Autoencoder",
        "model_type": str(cfg.autoencoder.model_type),
        "epochs": epochs,
        "latent_dim": cfg.dataset.latent_size,
        "learning_rate": cfg.training.learning_rate,
        "beta_start": cfg.training.beta_start,
        "beta_end": cfg.training.beta_end,
        "beta_anneal_epochs": cfg.training.beta_anneal_epochs,
        "seed": seed,
        "base_channels": cfg.autoencoder.base_channels,
        "num_blocks": cfg.autoencoder.num_blocks,
    }

    print("W&B Config:", wandb_cfg)

    wandb.init(
        entity="jmartini-tu-darmstadt",
        project="master-thesis",
        name=name,
        config=wandb_cfg,
        mode=wandb_mode,
    )

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=cfg.training.batch_size
    )

    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.training.learning_rate)

    loss_fn = nn.MSELoss()
    # loss_fn = HybridLoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = lpips.LPIPS(net="vgg").to(device)

    train_autoencoder(
        model=ae,
        device=device,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        beta_start=cfg.training.beta_start,
        beta_end=cfg.training.beta_end,
        beta_anneal_epochs=cfg.training.beta_anneal_epochs,
        rtpt=rtpt,
    )

    checkpoint_path = Path(f"checkpoints/{dataset_name}/autoencoder.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    ae_artifact = wandb.Artifact(name=name, type="autoencoder", metadata=wandb_cfg)
    ae_artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(ae_artifact)

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

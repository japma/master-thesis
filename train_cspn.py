"""Training entrypoint for CSPN."""

import torch
import tqdm
from pathlib import Path
from rtpt import RTPT
from torchinfo import summary
import wandb

from config import load_config
from models.autoencoder import VariationalAutoencoder, AbstractAutoencoder
from models.cspn import AbstractCSPN
from models.cspn.einet import Einet
from dataset_loaders import build_data_loaders
from losses import negative_log_likelihood_loss
from utils import seed_everything, resolve_device, load_checkpoint


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
        total_train_loss = torch.tensor(0.0, device=device)
        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                latent = autoencoder.encode(images)

            outputs = model(latent, labels)
            loss = negative_log_likelihood_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                outputs = model(latent, labels)
                loss = negative_log_likelihood_loss(outputs)
                # Accumulate without .item()
                total_val_loss += loss.detach()

        if epoch % 10 == 9 or epoch == 0:
            num_classes = 10
            samples_per_class = 3
            sample_labels = torch.arange(num_classes, device=device).repeat(
                samples_per_class
            )

            with torch.no_grad():
                sampled_latent = model.sample(sample_labels)
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


def main():
    cfg = load_config()

    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_cfg = cfg.dataset
    dataset_name = dataset_cfg.name
    epochs = cfg.training.epochs
    wandb_mode = cfg.wandb

    name = f"CSPN_{dataset_name}_seed{seed}"

    print(f"Training CSPN on {dataset_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)
    ae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        base_channels=cfg.autoencoder.base_channels,
        num_blocks=cfg.autoencoder.num_blocks,
        res_blocks=cfg.autoencoder.res_blocks,
    )

    ae_ckpt_path = Path(f"checkpoints/{dataset_name}/autoencoder.pt")
    if not ae_ckpt_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found at {ae_ckpt_path}. ")
    ae_ckpt = load_checkpoint(ae_ckpt_path, device)
    ae.load_state_dict(ae_ckpt)
    print(f"Loaded autoencoder from {ae_ckpt_path}")

    cspn = Einet(
        num_vars=cfg.dataset.latent_size,
        context_dim=cfg.dataset.num_classes,
        num_leaves=cfg.cspn.num_leaves,
        num_nodes=cfg.cspn.num_nodes,
        nn_hidden_dim=cfg.cspn.nn_hidden_dim,
        nn_num_hidden_layers=cfg.cspn.nn_num_hidden_layers,
    )

    print("CSPN Architecture:")
    summary(cspn)

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=name,
        max_iterations=max(epochs, 1),
    )
    rtpt.start()

    wandb_cfg = {
        "dataset": dataset_name,
        "model": "CSPN",
        "epochs": epochs,
        "latent_dim": cfg.dataset.latent_size,
        "learning_rate": cfg.training.learning_rate,
        "seed": seed,
        "num_leaves": cfg.cspn.num_leaves,
        "num_nodes": cfg.cspn.num_nodes,
        "nn_hidden_dim": cfg.cspn.nn_hidden_dim,
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

    optimizer = torch.optim.Adam(cspn.parameters(), lr=cfg.training.learning_rate)

    train_cspn(
        model=cspn,
        autoencoder=ae,
        device=device,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        rtpt=rtpt,
    )

    checkpoint_path = Path(f"checkpoints/{dataset_name}/cspn.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cspn.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    cspn_artifact = wandb.Artifact(name=name, type="cspn", metadata=wandb_cfg)
    cspn_artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(cspn_artifact)

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

from pathlib import Path

import torch
from hydra import main
from omegaconf import DictConfig
from rtpt import RTPT
import wandb
from dataset_loaders import build_data_loaders

from inference import run_inference
from utils import seed_everything, load_checkpoint, save_checkpoint
from utils.models import build_autoencoder, build_cspn
from utils.train import resolve_device
from train import train_autoencoder, train_cspn


@main(version_base=None, config_path="configs", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    seed = seed_everything(cfg.seed)

    dataset_cfg = cfg.dataset
    model_name = "Autoencoder" if cfg.mode == "train_ae" else "CSPN"
    dataset_name = dataset_cfg.name
    name = f"{model_name}_{dataset_name}_seed{seed}"
    device = resolve_device()
    epochs = cfg.training.epochs

    if cfg.mode == "train_ae" or cfg.mode == "train_cspn":
        rtpt = RTPT(
            name_initials="JM",
            experiment_name=name,
            max_iterations=max(cfg.training.epochs, 1),
        )
        rtpt.start()

        wandb_cfg = {
            "dataset": dataset_name,
            "model": model_name,
            "epochs": epochs,
            "latent_dim": cfg.dataset.latent_size,
            "learning_rate": cfg.training.learning_rate,
            "seed": seed,
        }

        print(wandb_cfg)

        wandb_run = wandb.init(
            entity="jmartini-tu-darmstadt",
            project="master-thesis",
            name=name,
            config=wandb_cfg,
            mode="online",
        )

        ae = build_autoencoder(cfg, device)
        train_loader, test_load = build_data_loaders(
            dataset_cfg, batch_size=cfg.training.batch_size
        )

        if cfg.mode == "train_ae":
            print("Training Autoencoder")
            optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.training.learning_rate)
            train_autoencoder(
                model=ae,
                device=device,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_load,
                optimizer=optimizer,
                beta=1.0,
                rtpt=rtpt,
            )

            checkpoint_path = "ae.pt"
            torch.save(ae.state_dict(), checkpoint_path)
            ae_artifact = wandb.Artifact(
                name=name, type="autoencoder", metadata=wandb_cfg
            )
            ae_artifact.add_file(checkpoint_path)
            wandb.log_artifact(ae_artifact)

        elif cfg.mode == "train_cspn":
            print("Training CSPN")
            cspn = build_cspn(cfg, device)
            # ae = load_checkpoint()
            optimizer = torch.optim.Adam(
                cspn.parameters(), lr=cfg.training.learning_rate
            )
            train_cspn(
                model=cspn,
                autoencoder=ae,
                device=device,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_load,
                optimizer=optimizer,
                rtpt=rtpt,
            )

        wandb.finish()
    elif cfg.mode == "inference":
        run_inference(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main_hydra()

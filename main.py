from models.cspn.einet import Einet
from models.autoencoder import VariationalAutoencoder
import lpips
from torchinfo import summary

from inference import run_cspn_inference, save_combined_latent_umap
from inference import run_ae_inference
from pathlib import Path

import torch
from rtpt import RTPT
import wandb
from dataset_loaders import build_data_loaders

from utils import seed_everything, load_checkpoint
from utils import resolve_device


def main_hydra(cfg) -> None:
    raise NotImplementedError
    seed = seed_everything(cfg.seed)

    dataset_cfg = cfg.dataset
    model_name = "Autoencoder" if cfg.mode == "train_ae" else "CSPN"
    dataset_name = dataset_cfg.name
    name = f"{model_name}_{dataset_name}_seed{seed}"
    device = resolve_device()
    epochs = cfg.training.epochs
    wandb_mode = cfg.wandb_mode

    if cfg.mode == "train_ae" or cfg.mode == "train_cspn" or cfg.mode == "inference_ae":
        ae_cfg = cfg.autoencoder
        input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

        ae = VariationalAutoencoder(
            input_shape=input_shape,
            latent_size=cfg.dataset.latent_size,
            base_channels=ae_cfg.base_channels,
            num_blocks=ae_cfg.num_blocks,
            res_blocks=ae_cfg.res_blocks,
        )
    else:
        ae = None

    if cfg.mode == "train_cspn" or cfg.mode == "inference_cspn":
        cspn = Einet(
            num_vars=cfg.dataset.latent_size,
            context_dim=cfg.dataset.num_classes,
            num_leaves=cfg.cspn.num_leaves,
            num_nodes=cfg.cspn.num_nodes,
            nn_hidden_dim=cfg.cspn.nn_hidden_dim,
            nn_num_hidden_layers=cfg.cspn.nn_num_hidden_layers,
        )
    else:
        cspn = None

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
            "beta_start": cfg.training.beta_start,
            "beta_end": cfg.training.beta_end,
            "beta_anneal_epochs": cfg.training.beta_anneal_epochs,
            "seed": seed,
        }

        print(wandb_cfg)

        wandb.init(
            entity="jmartini-tu-darmstadt",
            project="master-thesis",
            name=name,
            config=wandb_cfg,
            mode=wandb_mode,
        )

        train_loader, test_load, (train_dataset, test_dataset) = build_data_loaders(
            dataset_cfg, batch_size=cfg.training.batch_size
        )

        if cfg.mode == "train_ae":
            print("Training Autoencoder")
            if ae is None:
                raise ValueError("Autoencoder model not initialized")
            optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.training.learning_rate)
            train_autoencoder(
                model=ae,
                device=device,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_load,
                optimizer=optimizer,
                loss_fn=lpips.LPIPS(net="vgg").to(device),
                # loss_fn=nn.MSELoss(),
                beta_start=cfg.training.beta_start,
                beta_end=cfg.training.beta_end,
                beta_anneal_epochs=cfg.training.beta_anneal_epochs,
                rtpt=rtpt,
            )

            checkpoint_path = Path(f"checkpoints/{dataset_name}/autoencoder.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ae.state_dict(), checkpoint_path)
            ae_artifact = wandb.Artifact(
                name=name, type="autoencoder", metadata=wandb_cfg
            )
            ae_artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(ae_artifact)

        elif cfg.mode == "train_cspn":
            print("Training CSPN")
            if ae is None:
                raise ValueError("Autoencoder model not initialized")
            if cspn is None:
                raise ValueError("CSPN model not initialized")
            summary(cspn)

            ae_ckpt = load_checkpoint(Path("checkpoints/MNIST/autoencoder.pt"), device)
            ae.load_state_dict(ae_ckpt)
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
            checkpoint_path = Path(f"checkpoints/{dataset_name}/cspn.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cspn.state_dict(), checkpoint_path)
            cspn_artifact = wandb.Artifact(name=name, type="cspn", metadata=wandb_cfg)
            cspn_artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(cspn_artifact)

        wandb.finish()
    elif cfg.mode == "inference_ae":
        if ae is None:
            raise ValueError("Autoencoder model not initialized")
        ae_ckpt = load_checkpoint(
            Path(f"checkpoints/{dataset_name}/autoencoder.pt"), device
        )
        ae.load_state_dict(ae_ckpt)
        _, test_loader, (_, test_dataset) = build_data_loaders(
            dataset_cfg, batch_size=cfg.training.batch_size
        )
        run_ae_inference(model=ae, data_loader=test_loader, device=device)
    elif cfg.mode == "inference_cspn":
        if cspn is None:
            raise ValueError("CSPN model not initialized")
        cspn_ckpt = load_checkpoint(Path(f"checkpoints/{dataset_name}/cspn.pt"), device)
        cspn.load_state_dict(cspn_ckpt)

        if ae is None:
            # TODO fix this
            ae = build_autoencoder(cfg, device)
        ae_ckpt = load_checkpoint(
            Path(f"checkpoints/{dataset_name}/autoencoder.pt"), device
        )
        ae.load_state_dict(ae_ckpt)

        _, test_loader, (_, test_dataset) = build_data_loaders(
            dataset_cfg, batch_size=cfg.training.batch_size
        )

        class_names = getattr(test_dataset, "class_names", None)
        cspn_latents, cspn_labels = run_cspn_inference(
            model=cspn,
            data_loader=None,
            device=device,
            autoencoder=ae,
            class_names=class_names,
        )

        ae_latents, ae_labels = run_ae_inference(
            model=ae, data_loader=test_loader, device=device
        )

        # Create combined visualization
        save_combined_latent_umap(
            ae_latents=ae_latents,
            cspn_latents=cspn_latents,
            ae_labels=ae_labels,
            cspn_labels=cspn_labels,
            path="combined_ae_cspn_umap.png",
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main_hydra()

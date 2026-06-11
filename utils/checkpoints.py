from pathlib import Path
import torch
from models.autoencoder import AbstractAutoencoder, VariationalAutoencoder
from models.cspn import AbstractCSPN
from models.cspn.einet import Einet


# --- Autoencoder ---
def save_autoencoder(model: AbstractAutoencoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )
    print("Saved autoencoder checkpoint to", path)


def _create_autoencoder_from_checkpoint(cfg: dict) -> AbstractAutoencoder:
    if cfg["model_type"] == "variational":
        return VariationalAutoencoder(
            input_shape=cfg["input_shape"],
            latent_dim=cfg["latent_dim"],
            base_channels=cfg["base_channels"],
            num_blocks=cfg["num_blocks"],
            res_blocks=cfg["res_blocks"],
        )

    else:
        raise ValueError(f"Unknown autoencoder type: {cfg['type']}")


def load_ae_from_path(path: Path, device=None) -> AbstractAutoencoder:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = _create_autoencoder_from_checkpoint(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model


# --- CSPN ---
def save_cspn(model: AbstractCSPN, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )
    print("Saved CSPN checkpoint to", path)


def _create_cspn_from_checkpoint(cfg: dict) -> AbstractCSPN:
    return Einet(
        num_vars=cfg["num_vars"],
        context_dim=cfg["context_dim"],
        num_leaves=cfg["num_leaves"],
        num_nodes=cfg["num_nodes"],
        nn_hidden_dim=cfg["nn_hidden_dim"],
        nn_num_hidden_layers=cfg["nn_num_hidden_layers"],
    )


def load_cspn_from_path(path: Path, device=None) -> AbstractCSPN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = _create_cspn_from_checkpoint(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model

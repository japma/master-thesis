from pathlib import Path

import torch

import wandb
from models.autoencoder import (
    AbstractAutoencoder,
    AutoencoderType,
    VariationalAutoencoder,
)
from models.cspn import AbstractCSPN
from models.cspn.abstract_cspn import CSPNType
from models.cspn.CustomEinet.einet import Einet
from models.cspn.psinet_cspn import PsiNetCSPN
from models.cspn.spflow_cspn import SPFlowCSPN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# --- General ---
def load_from_wandb(
    ckpt_name: str,
    tag: str = "latest",
) -> Path:
    entity = "jmartini-tu-darmstadt"
    project = "master-thesis"
    name = f"{entity}/{project}/{ckpt_name}:{tag}"
    api = wandb.Api()
    artifact = api.artifact(name)
    file = artifact.file(str(ARTIFACTS_DIR))
    print(f"Loading {file} from Weights & Biases artifact {name}")
    return Path(file)


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
    match cfg["model_type"]:
        case AutoencoderType.VARIATIONAL:
            return VariationalAutoencoder(
                input_shape=cfg["input_shape"],
                latent_dim=cfg["latent_dim"],
                base_channels=cfg["base_channels"],
                num_blocks=cfg["num_blocks"],
                res_blocks=cfg["res_blocks"],
            )
        case _:
            raise ValueError(f"Unknown autoencoder type: {cfg['model_type']}")


def load_ae_from_path(path: Path, device=None) -> AbstractAutoencoder:
    with torch.serialization.safe_globals([AutoencoderType]):
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
    match cfg["model_type"]:
        case CSPNType.CUSTOM | CSPNType.CUSTOM_DEPRECATED:
            return Einet(
                num_vars=cfg["num_vars"],
                context_dim=cfg["context_dim"],
                num_leaves=cfg["num_leaves"],
                num_nodes=cfg["num_nodes"],
                nn_hidden_dim=cfg["nn_hidden_dim"],
                nn_num_hidden_layers=cfg["nn_num_hidden_layers"],
            )
        case CSPNType.SPFLOW:
            return SPFlowCSPN(
                latent_dim=cfg["latent_dim"],
                num_classes=cfg["num_classes"],
                num_sums=cfg["num_sums"],
                num_leaves=cfg["num_leaves"],
                depth=cfg["depth"],
                num_repetitions=cfg["num_repetitions"],
                nn_layers=cfg["nn_layers"],
                nn_hidden_dim=cfg["nn_hidden_dim"],
            )
        case CSPNType.PSINET | CSPNType.PSINET_DEPRECATED:
            return PsiNetCSPN(
                latent_dim=cfg["latent_dim"],
                num_classes=cfg["num_classes"],
                num_repetitions=5,
                num_input_distributions=10,
                num_sums=10,
                min_var=1.0,
                max_var=2.0,
                h_dims=[128, 128],
            )
        case _:
            raise ValueError(f"Unknown CSPN type: {cfg['model_type']}")


def load_cspn_from_path(path: Path, device=None) -> AbstractCSPN:
    with torch.serialization.safe_globals([CSPNType]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model = _create_cspn_from_checkpoint(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model

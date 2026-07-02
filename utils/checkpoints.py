from pathlib import Path

import torch

import wandb
from models.autoencoder import (
    AbstractAutoencoder,
    VariationalAutoencoder,
)
from models.cspn import AbstractCSPN
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import AutoencoderConfig, AutoencoderType, CSPNConfig, CSPNType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# --- General ---
def load_from_wandb(
    ckpt_name: str,
    tag: str = "latest",
) -> Path:
    """Load a checkpoint from wandb artifacts. Uses the most recent checkpoint unless tag is provided."""
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


def _create_autoencoder_from_checkpoint(cfg: AutoencoderConfig) -> AbstractAutoencoder:
    return VariationalAutoencoder(config=cfg)


def load_ae_from_path(path: Path, device=None) -> AbstractAutoencoder:
    with torch.serialization.safe_globals([AutoencoderType]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = AutoencoderConfig.model_validate(ckpt["model_cfg"])
    model = _create_autoencoder_from_checkpoint(cfg)
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


def _create_cspn_from_checkpoint(cfg: CSPNConfig) -> AbstractCSPN:
    return PsiNetCSPN(config=cfg)


def load_cspn_from_path(path: Path, device=None) -> AbstractCSPN:
    with torch.serialization.safe_globals([CSPNType]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = CSPNConfig.model_validate(ckpt["model_cfg"])
    model = _create_cspn_from_checkpoint(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model

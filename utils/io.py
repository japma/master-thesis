"""Helpers for organizing run output directories and saved checkpoints."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunDirectories:
    """Directories created for a single training or inference run."""

    run_dir: Path
    checkpoints_dir: Path
    images_dir: Path


def create_run_directories(run_dir: str | Path) -> RunDirectories:
    """Create the per-run directory structure used for saved artifacts."""
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    images_dir = run_dir / "images"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return RunDirectories(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        images_dir=images_dir,
    )


# TODO rework this to use model directly
def save_checkpoint(model_state_dict, checkpoints_dir: Path, name: str) -> Path:
    save_path = checkpoints_dir / f"{name}.pt"
    torch.save(model_state_dict, save_path)
    logger.info("Checkpoint saved to %s", save_path)
    return save_path


def load_checkpoint(load_path: Path, map_location=None):
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    logger.info("Loading checkpoint from %s", load_path)
    return torch.load(load_path, map_location=map_location, weights_only=True)


def build_ae_path(cfg: DictConfig) -> Path:
    if cfg.paths.ae_checkpoint is not None:
        return cfg.paths.ae_checkpoint
    else:
        path = f"checkpoints/{cfg.dataset.name}/autoencoder.pt"
        return Path(path)

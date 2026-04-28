"""Helpers for organizing run output directories and saved checkpoints."""

from dataclasses import dataclass
from pathlib import Path

import torch


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


def save_checkpoint(model_state_dict, checkpoints_dir: Path, name: str) -> Path:
    save_path = checkpoints_dir / f"{name}.pt"
    torch.save(model_state_dict, save_path)
    print(f"\nCheckpoint saved to {save_path}")
    return save_path


def load_checkpoint(checkpoints_dir: Path, name: str, map_location=None):
    load_path = checkpoints_dir / f"{name}.pt"
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    print(f"Loading checkpoint from {load_path}")
    return torch.load(load_path, map_location=map_location, weights_only=True)

"""Helpers for organizing run output directories and saved checkpoints."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass(frozen=True)
class RunDirectories:
    """Directories created for a single training run."""

    run_dir: Path
    checkpoints_dir: Path
    images_dir: Path


def create_run_directories(output_dir: str, dataset_name: str) -> RunDirectories:
    """Create the per-run directory structure used for saved artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{dataset_name.lower()}_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    images_dir = run_dir / "images"

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

from pathlib import Path

import torch


def load_checkpoint(load_path: Path, map_location=None):
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    return torch.load(load_path, map_location=map_location, weights_only=True)

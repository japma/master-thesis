import os
import random
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(load_path: Path, map_location=None):
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    return torch.load(load_path, map_location=map_location, weights_only=True)


def _normalize_seed(seed) -> int:
    if seed is None or seed == "":
        return random.SystemRandom().randint(0, 2**32 - 1)

    if isinstance(seed, bool):
        raise ValueError("seed must be an integer or null")

    try:
        seed_value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer or null") from exc

    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")

    return seed_value % 2**32


def seed_everything(seed=None) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed to use. If None, a random seed is generated.

    Returns:
        The seed that was applied.
    """
    seed_value = _normalize_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed_value)

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed_value


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

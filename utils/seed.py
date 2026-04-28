"""Random seeding helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def _normalize_seed(seed: Any) -> int:
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


def seed_everything(seed: Any | None = None) -> int:
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

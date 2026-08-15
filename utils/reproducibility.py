import os
import random

import numpy as np
import torch


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
    seed_value = _normalize_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed_value)

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return seed_value


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_rng_state() -> dict:
    """Snapshot of every RNG seed_everything touches, for exact-resume checkpointing."""
    state: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    """Inverse of get_rng_state(); restores RNG state saved by a previous run."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])

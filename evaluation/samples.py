"""What every stage agrees on about a sample: its label columns and its pixel format."""

import subprocess

import torch

# Column of a colour-MNIST label row each factor lives in.
DIGIT, FG, BG = 0, 1, 2


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Decoder output in [0, 1] as the 8-bit image it is actually standing in for."""
    return (images.clamp(0, 1) * 255).round().to(torch.uint8)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_uint8`, back into the [0, 1] range every model here expects."""
    return images.float() / 255.0


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None

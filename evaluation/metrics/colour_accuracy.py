"""Does the image carry the foreground and background colour it was asked for?

One row per factor, read off the pixels against the fixed palette.
"""

import pandas as pd
import torch

from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.samples import BG, FG

FILENAME = "colour_accuracy.csv"


def correct(images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per image and per factor: 1.0 where the nearest palette entry is the target."""
    return {
        "fg": (
            nearest_palette_index(foreground_colour(images), FG_PALETTE)
            == labels[:, FG]
        ).float(),
        "bg": (
            nearest_palette_index(border_colour(images), BG_PALETTE) == labels[:, BG]
        ).float(),
    }


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"factor": factor, "value": float(hits.mean()), "n": int(hits.shape[0])}
            for factor, hits in correct(images, labels).items()
        ]
    )

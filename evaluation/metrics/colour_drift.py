"""RGB distance from the colour that was asked for. Keeps moving once accuracy is 1.0."""

import pandas as pd
import torch

from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    palette_colour,
)
from evaluation.samples import BG, FG

FILENAME = "colour_drift.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    distances = {
        "fg": (
            foreground_colour(images) - palette_colour(FG_PALETTE, labels[:, FG])
        ).norm(dim=1),
        "bg": (border_colour(images) - palette_colour(BG_PALETTE, labels[:, BG])).norm(
            dim=1
        ),
    }
    return pd.DataFrame(
        [
            {"factor": factor, "value": float(d.mean()), "n": int(d.shape[0])}
            for factor, d in distances.items()
        ]
    )

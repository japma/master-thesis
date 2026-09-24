"""Digit accuracy per (digit, fg, bg) cell: one row per cell present in `labels`."""

import pandas as pd
import torch

from evaluation.samples import BG, DIGIT, FG

FILENAME = "digit_accuracy_by_combination.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "digit": labels[:, DIGIT].numpy(),
            "fg": labels[:, FG].numpy(),
            "bg": labels[:, BG].numpy(),
            "correct": (predictions[:, DIGIT] == labels[:, DIGIT]).float().numpy(),
        }
    )
    return frame.groupby(["digit", "fg", "bg"], as_index=False).agg(
        value=("correct", "mean"), n=("correct", "size")
    )

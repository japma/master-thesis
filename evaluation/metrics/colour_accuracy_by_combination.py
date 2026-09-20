"""Colour accuracy per (digit, fg, bg) cell: the held-out combinations are read here."""

import pandas as pd
import torch

from evaluation.metrics.colour_accuracy import correct
from evaluation.samples import BG, DIGIT, FG

FILENAME = "colour_accuracy_by_combination.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    cells = []
    for factor, hits in correct(images, labels).items():
        frame = pd.DataFrame(
            {
                "digit": labels[:, DIGIT].numpy(),
                "fg": labels[:, FG].numpy(),
                "bg": labels[:, BG].numpy(),
                "correct": hits.numpy(),
            }
        )
        grouped = frame.groupby(["digit", "fg", "bg"], as_index=False).agg(
            value=("correct", "mean"), n=("correct", "size")
        )
        cells.append(grouped.assign(factor=factor))
    table = pd.concat(cells, ignore_index=True)
    return table[["factor", "digit", "fg", "bg", "value", "n"]]

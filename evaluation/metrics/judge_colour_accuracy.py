"""Does the judge read the foreground and background colour that was asked for?

The learned counterpart of `colour_accuracy`, which reads colour off the pixels against
the palette. One row per factor.
"""

import pandas as pd
import torch

from evaluation.samples import BG, FG

FILENAME = "judge_colour_accuracy.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor": factor,
                "value": float(
                    (predictions[:, column] == labels[:, column]).float().mean()
                ),
                "n": int(labels.shape[0]),
            }
            for factor, column in (("fg", FG), ("bg", BG))
        ]
    )

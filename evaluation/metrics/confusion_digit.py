"""One row per (truth, prediction) digit pair, zeros included so the grid is complete."""

import pandas as pd
import torch

from evaluation.samples import DIGIT

FILENAME = "confusion_digit.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    truth_labels = labels[:, DIGIT]
    counts = torch.bincount(
        truth_labels * num_classes + predictions, minlength=num_classes**2
    )
    truth, predicted = torch.meshgrid(
        torch.arange(num_classes), torch.arange(num_classes), indexing="ij"
    )
    return pd.DataFrame(
        {
            "truth": truth.flatten().numpy(),
            "predicted": predicted.flatten().numpy(),
            "n": counts.numpy(),
        }
    )

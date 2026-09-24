"""Share of images the judge reads as the digit they were asked for."""

import pandas as pd
import torch

from evaluation.samples import DIGIT

FILENAME = "digit_accuracy.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    truth = labels[:, DIGIT]
    correct = float((predictions[:, DIGIT] == truth).float().mean())
    return pd.DataFrame([{"value": correct, "n": int(truth.shape[0])}])

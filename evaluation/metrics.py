"""The numbers. Predictions and labels in, one value or one small table out."""

import pandas as pd
import torch

# Colour-MNIST labels are [digit, fg, bg].
DIGIT, FG, BG = 0, 1, 2


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((predictions == targets).float().mean())


def accuracy_by_combination(
    predictions: torch.Tensor, labels: torch.Tensor, factor: int = DIGIT
) -> pd.DataFrame:
    """Accuracy per (digit, fg, bg) cell: one row per cell present in `labels`."""
    frame = pd.DataFrame(
        {
            "digit": labels[:, DIGIT].numpy(),
            "fg": labels[:, FG].numpy(),
            "bg": labels[:, BG].numpy(),
            "correct": (predictions == labels[:, factor]).float().numpy(),
        }
    )
    return frame.groupby(["digit", "fg", "bg"], as_index=False).agg(
        value=("correct", "mean"), n=("correct", "size")
    )


def confusion(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> pd.DataFrame:
    """One row per (truth, prediction) pair, zeros included so the grid is complete."""
    counts = torch.bincount(
        targets * num_classes + predictions, minlength=num_classes**2
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

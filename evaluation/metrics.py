"""The judged metrics: one function each, and `METRICS`, which names the CSV each writes.

Every function takes the judge's `predictions` and the `labels` they are read against,
both `(N, len(factors))` and restricted to the factors the set was conditioned on, and
returns its own table with a `factor` column.
"""

from collections.abc import Callable

import pandas as pd
import torch


def accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    factors: list[str],
    cardinalities: list[int],
) -> pd.DataFrame:
    """Share of images the judge reads as the class that was asked for."""
    correct = (predictions == labels).float().mean(dim=0)
    return pd.DataFrame(
        {"factor": factors, "value": correct.tolist(), "n": labels.shape[0]}
    )


def accuracy_by_combination(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    factors: list[str],
    cardinalities: list[int],
) -> pd.DataFrame:
    """Accuracy per label combination, one row per factor and cell present in `labels`:
    the held-out cells are read here."""
    cells = pd.DataFrame(labels.numpy(), columns=factors)
    correct = (predictions == labels).float().numpy()
    tables = [
        cells.assign(correct=correct[:, i])
        .groupby(factors, as_index=False)
        .agg(value=("correct", "mean"), n=("correct", "size"))
        .assign(factor=factor)
        for i, factor in enumerate(factors)
    ]
    return pd.concat(tables, ignore_index=True)[["factor", *factors, "value", "n"]]


def confusion(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    factors: list[str],
    cardinalities: list[int],
) -> pd.DataFrame:
    """Counts per factor and (truth, predicted) pair, zeros included so every grid is
    complete."""
    tables = []
    for i, (factor, classes) in enumerate(zip(factors, cardinalities, strict=True)):
        counts = torch.bincount(
            labels[:, i] * classes + predictions[:, i], minlength=classes**2
        )
        truth, predicted = torch.meshgrid(
            torch.arange(classes), torch.arange(classes), indexing="ij"
        )
        tables.append(
            pd.DataFrame(
                {
                    "factor": factor,
                    "truth": truth.flatten().numpy(),
                    "predicted": predicted.flatten().numpy(),
                    "n": counts.numpy(),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


Metric = Callable[[torch.Tensor, torch.Tensor, list[str], list[int]], pd.DataFrame]

# Each writes `<name>.csv`; a config's `evaluation.metrics` selects by these names.
METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "accuracy_by_combination": accuracy_by_combination,
    "confusion": confusion,
}

"""What the training split says a factor's distribution is.

The ground truth a marginalized query is scored against. Counted off the labels the
model was actually trained on, not read from the weight table: the table is the intent,
the labels are what happened.
"""

import math
from collections.abc import Sequence

import torch

from dataset_loaders import build_dataset
from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG

# A query is a label row with UNSPECIFIED where a factor is free to vary.
UNSPECIFIED = -1

CARDINALITIES = (NUM_DIGITS, NUM_FG, NUM_BG)


def training_labels(dataset: str) -> torch.Tensor:
    """The `(N, factors)` labels the model was trained on.

    Colour-MNIST calls them `targets`, torchvision's CelebA calls them `attr`.
    """
    data = build_dataset(dataset, train=True)
    for attribute in ("targets", "attr"):
        labels = getattr(data, attribute, None)
        if labels is not None:
            return torch.as_tensor(labels).long()
    raise AttributeError(
        f"{dataset} exposes neither `targets` nor `attr`, so its labels cannot be read"
    )


def matching(labels: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the rows of `labels` a query's specified factors select."""
    rows = torch.ones(labels.shape[0], dtype=torch.bool)
    for factor, value in enumerate(query.tolist()):
        if value != UNSPECIFIED:
            rows &= labels[:, factor] == value
    return rows


def cell_index(values: torch.Tensor, factors: Sequence[int]) -> torch.Tensor:
    """Flat index of each row's values over `factors`, row-major in the order given."""
    index = torch.zeros(values.shape[0], dtype=torch.long)
    for column, factor in enumerate(factors):
        index = index * CARDINALITIES[factor] + values[:, column]
    return index


def num_cells(factors: Sequence[int]) -> int:
    return math.prod(CARDINALITIES[f] for f in factors)


def conditional(
    labels: torch.Tensor, query: torch.Tensor, factors: Sequence[int]
) -> torch.Tensor:
    """p(factors | the query's specified factors), flattened as `cell_index` lays it out."""
    rows = matching(labels, query)
    if not bool(rows.any()):
        raise ValueError(
            f"no training rows match query {query.tolist()}, so it has no conditional "
            "to be scored against -- that combination was held out"
        )
    cells = cell_index(labels[rows][:, list(factors)], factors)
    counts = torch.bincount(cells, minlength=num_cells(factors))
    return counts / counts.sum()


def total_variation(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """Half the L1 distance between two distributions: 0 identical, 1 disjoint."""
    return float(0.5 * (generated - truth).abs().sum())

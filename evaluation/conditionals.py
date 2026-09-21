"""What the training split says a factor's distribution is.

The ground truth a marginalized query is scored against. Counted off the labels the
model was actually trained on, not read from the weight table: the table is the intent,
the labels are what happened.
"""

import torch

from dataset_loaders import build_dataset
from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG

# A query is a label row with UNSPECIFIED where a factor is free to vary.
UNSPECIFIED = -1

FACTOR_NAMES = ("digit", "fg", "bg")
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


def conditional(labels: torch.Tensor, query: torch.Tensor, factor: int) -> torch.Tensor:
    """p(factor | the query's specified factors), as a `(cardinality,)` tensor."""
    rows = matching(labels, query)
    if not bool(rows.any()):
        raise ValueError(
            f"no training rows match query {query.tolist()}, so it has no conditional "
            "to be scored against -- that combination was held out"
        )
    counts = torch.bincount(labels[rows, factor], minlength=CARDINALITIES[factor])
    return counts / counts.sum()


def total_variation(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """Half the L1 distance between two distributions: 0 identical, 1 disjoint."""
    return float(0.5 * (generated - truth).abs().sum())

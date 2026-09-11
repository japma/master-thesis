"""The fixed latent coordinates each label value is pinned to."""

import torch

from dataset_loaders.colour_mnist import BG_COLOURS, FG_COLOURS
from utils.config import AnchorScheme, SupervisionConfig


def _palette(colours: dict[str, tuple[int, int, int]]) -> torch.Tensor:
    return torch.tensor(list(colours.values()), dtype=torch.float32) / 255.0


def anchor_table(scheme: AnchorScheme, cardinality: int, dim: int) -> torch.Tensor:
    match scheme:
        case AnchorScheme.NONE:
            raise ValueError("AnchorScheme.NONE has no anchor table")
        case AnchorScheme.ONEHOT:
            table = torch.eye(cardinality)
        case AnchorScheme.COLOUR_MNIST_FG:
            table = _palette(FG_COLOURS)
        case AnchorScheme.COLOUR_MNIST_BG:
            table = _palette(BG_COLOURS)

    if table.shape != (cardinality, dim):
        raise ValueError(
            f"anchor scheme {scheme} needs a block of shape "
            f"(cardinality={table.shape[0]}, dims={table.shape[1]}), but the factor "
            f"declares (cardinality={cardinality}, dims={dim})"
        )
    return table


def anchor_tables(supervision: SupervisionConfig) -> list[torch.Tensor]:
    """One table per anchored factor, in target-vector order."""
    return [
        anchor_table(
            supervision.anchor_schemes[i],
            supervision.cardinalities[i],
            supervision.dims[i],
        )
        for i in supervision.anchored_factors
    ]


def anchor_targets(
    tables: list[torch.Tensor], factor_ids: list[int], labels: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [table[labels[:, i]] for table, i in zip(tables, factor_ids, strict=True)],
        dim=1,
    )

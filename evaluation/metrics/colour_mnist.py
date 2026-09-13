"""Colour metrics, specific to colour-MNIST and deliberately not generalized."""

import torch

from evaluation.batch import EvalBatch
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.harness import Metric, MetricResult, PerCombination


class ColourFidelity(Metric):
    name = "colour"
    requires = frozenset({"images"})

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    def update(self, batch: EvalBatch) -> None:
        assert batch.images is not None
        images = batch.images
        labels = batch.labels.cpu().numpy()
        locate_in = batch.reference if batch.reference is not None else images

        generated_bg = border_colour(images)
        generated_fg = foreground_colour(locate_in, images)

        target_bg = torch.tensor(
            BG_PALETTE[labels[:, 2]], dtype=images.dtype, device=images.device
        )
        target_fg = torch.tensor(
            FG_PALETTE[labels[:, 1]], dtype=images.dtype, device=images.device
        )

        self.accumulator.add(
            batch.index,
            bg_accuracy=nearest_palette_index(generated_bg, BG_PALETTE) == labels[:, 2],
            fg_accuracy=nearest_palette_index(generated_fg, FG_PALETTE) == labels[:, 1],
            bg_drift=(generated_bg - target_bg).norm(dim=1).cpu().numpy(),
            fg_drift=(generated_fg - target_fg).norm(dim=1).cpu().numpy(),
            contrast=(generated_fg - generated_bg).norm(dim=1).cpu().numpy(),
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())

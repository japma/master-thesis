"""Figures are evidence in the writeup, so what matters is that a cell in a grid really
is the combination its axis says it is -- a silent reshape would mislabel every picture
without changing a single number.
"""

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from utils.visualisation import (
    plot_combination_grid,
    plot_image_rows,
)


def test_image_rows_lays_out_one_axis_per_image() -> None:
    rows = {"original": torch.rand(4, 3, 8, 8), "recon": torch.rand(4, 3, 8, 8)}
    figure = plot_image_rows(rows, title="t", col_labels=list("abcd"))

    assert len(figure.axes) == 8
    assert figure.axes[0].get_title() == "a"


def test_image_rows_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_image_rows({"a": torch.rand(3, 3, 8, 8), "b": torch.rand(2, 3, 8, 8)})


def test_combination_grid_draws_every_cell() -> None:
    images = torch.rand(NUM_DIGITS, NUM_FG, NUM_BG, 3, 8, 8)
    figure = plot_combination_grid(images, title="samples")

    canvas = figure.axes[0].get_images()[0].get_array()
    assert canvas.shape[0] == NUM_DIGITS * (8 + 2) + 2
    assert canvas.shape[1] == NUM_FG * NUM_BG * (8 + 2) + 2
    assert len(figure.axes[0].get_yticklabels()) == NUM_DIGITS

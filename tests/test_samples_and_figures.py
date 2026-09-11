"""Figures are evidence in the writeup, so the thing that matters is that a cell in a
grid really is the combination its axis says it is -- a silent reshape would mislabel
every picture without changing a single number.
"""

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation.samples import (
    reconstruct,
    sample_combination_grid,
    sample_for_label,
)
from models.autoencoder import AbstractAutoencoder
from utils.visualisation import plot_combination_grid, plot_image_rows

IMAGE_SIZE = 4


class LabelEchoModel:
    """`sample` returns the label itself as a latent, so the picture that comes out the
    other end can be checked against the label that went in."""

    def __init__(self) -> None:
        self.std_corrections: list[float] = []

    def eval(self) -> None:
        pass

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        self.std_corrections.append(std_correction)
        return labels.float()


class ChannelPaintingAE(AbstractAutoencoder):
    """Paints latent[i] into channel i, so a decoded image carries its label."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :3, None, None].expand(-1, 3, IMAGE_SIZE, IMAGE_SIZE).clone()

    def get_latent_dim(self) -> torch.Size:
        return torch.Size([3])


def label_of(image: torch.Tensor) -> tuple[int, int, int]:
    return tuple(round(float(v)) for v in image[:, 0, 0])


def test_the_grid_cell_matches_its_combination() -> None:
    grid = sample_combination_grid(
        LabelEchoModel(), ChannelPaintingAE(), torch.device("cpu")
    )

    assert grid.shape == (NUM_DIGITS, NUM_FG, NUM_BG, 1, 3, IMAGE_SIZE, IMAGE_SIZE)
    for digit in (0, 4, 9):
        for fg in (0, 5):
            for bg in range(NUM_BG):
                assert label_of(grid[digit, fg, bg, 0]) == (digit, fg, bg)


def test_multiple_samples_per_combination_keep_the_layout() -> None:
    grid = sample_combination_grid(
        LabelEchoModel(),
        ChannelPaintingAE(),
        torch.device("cpu"),
        samples_per_combination=3,
    )

    assert grid.shape[:4] == (NUM_DIGITS, NUM_FG, NUM_BG, 3)
    for sample in range(3):
        assert label_of(grid[7, 2, 1, sample]) == (7, 2, 1)


def test_sample_for_label_repeats_one_combination() -> None:
    model = LabelEchoModel()
    images = sample_for_label(
        model, ChannelPaintingAE(), (3, 4, 2), count=5, device=torch.device("cpu"),
        std_correction=0.6,
    )

    assert images.shape == (5, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert {label_of(image) for image in images} == {(3, 4, 2)}
    assert model.std_corrections == [0.6]


def test_reconstruct_goes_through_the_posterior_mean() -> None:
    images = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = reconstruct(ChannelPaintingAE(), images, torch.device("cpu"))

    assert output.shape == images.shape
    assert torch.allclose(output[:, :, 0, 0], images.mean(dim=(2, 3)), atol=1e-6)


# --- figures ---
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


def test_latent_traversal_sweeps_one_dimension_per_row() -> None:
    from evaluation.samples import latent_traversal

    ae = ChannelPaintingAE()
    rows = latent_traversal(
        ae, torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), dims=[0, 2],
        values=[-1.0, 0.0, 1.0], device=torch.device("cpu"),
    )

    assert list(rows) == ["dim 0", "dim 2"]
    assert rows["dim 0"].shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    # The swept dimension takes the given values; the others stay where encoding put them.
    assert [round(float(v), 3) for v in rows["dim 0"][:, 0, 0, 0]] == [-1.0, 0.0, 1.0]
    assert len({round(float(v), 5) for v in rows["dim 0"][:, 1, 0, 0]}) == 1

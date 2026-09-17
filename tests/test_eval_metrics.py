"""The metric functions, checked on inputs whose answers are known."""

import math

import pytest
import torch
import torch.nn.functional as F

from dataset_loaders.colour_mnist import all_combinations, combination_index
from evaluation.colour import BG_PALETTE, FG_PALETTE, border_colour, foreground_colour
from evaluation.metrics import (
    colour_drift,
    confidence,
    contrast,
    digit_accuracy,
    entropy,
    factor_label_accuracy,
    fit_gaussian,
    joint_label_accuracy,
    mahalanobis,
    palette_accuracy,
    predicted_class_entropy,
    spread,
)

IMAGE = (3, 28, 28)
BLACK, GREEN = 1, 1  # indices into BG_PALETTE / FG_PALETTE


def painted(labels: torch.Tensor) -> torch.Tensor:
    """A centre square of each row's foreground colour on its background colour."""
    images = (
        torch.tensor(BG_PALETTE[labels[:, 2]], dtype=torch.float32)
        .reshape(-1, 3, 1, 1)
        .expand(-1, *IMAGE)
        .clone()
    )
    fg = torch.tensor(FG_PALETTE[labels[:, 1]], dtype=torch.float32)
    images[:, :, 10:18, 10:18] = fg.reshape(-1, 3, 1, 1)
    return images


def test_combination_index_matches_all_combinations() -> None:
    combinations = all_combinations()
    assert torch.equal(combination_index(combinations), torch.arange(180))


def test_colour_metrics_are_perfect_on_correctly_painted_images() -> None:
    labels = all_combinations()
    images = painted(labels)
    bg = border_colour(images)
    fg = foreground_colour(images, images)

    assert palette_accuracy(bg, BG_PALETTE, labels[:, 2]).min() == 1.0
    assert palette_accuracy(fg, FG_PALETTE, labels[:, 1]).min() == 1.0
    assert colour_drift(bg, BG_PALETTE, labels[:, 2]).max() < 1e-5
    assert contrast(fg, bg).min() > 0.0


def test_palette_accuracy_catches_a_swapped_background() -> None:
    labels = all_combinations()
    wrong = labels.clone()
    wrong[:, 2] = (wrong[:, 2] + 1) % 3

    bg = border_colour(painted(wrong))
    assert palette_accuracy(bg, BG_PALETTE, labels[:, 2]).max() == 0.0
    assert colour_drift(bg, BG_PALETTE, labels[:, 2]).min() > 0.1


def test_foreground_is_located_in_the_reference() -> None:
    """A reconstruction that moved the digit must not score well once the original is
    used to find the foreground pixels."""
    labels = torch.tensor([[0, GREEN, BLACK]])
    green = torch.tensor(FG_PALETTE[GREEN], dtype=torch.float32).reshape(1, 3, 1, 1)

    original = painted(torch.tensor([[0, 0, BLACK]]))
    original[:, :, 9:19, 9:19] = green
    misplaced = painted(torch.tensor([[0, 0, BLACK]]))
    misplaced[:, :, 9:19, 9:19] = torch.tensor(BG_PALETTE[BLACK]).reshape(1, 3, 1, 1)
    misplaced[:, :, 22:26, 22:26] = green

    own = foreground_colour(misplaced, misplaced)
    assert palette_accuracy(own, FG_PALETTE, labels[:, 1]).item() == 1.0

    located = foreground_colour(original, misplaced)
    assert palette_accuracy(located, FG_PALETTE, labels[:, 1]).item() == 0.0
    assert colour_drift(located, FG_PALETTE, labels[:, 1]).item() > 0.5


def test_digit_metrics_read_the_logits() -> None:
    digits = torch.arange(10)
    certain = F.one_hot(digits, 10).float() * 20.0
    uniform = torch.zeros(10, 10)

    assert digit_accuracy(certain, digits).min() == 1.0
    assert (
        digit_accuracy(F.one_hot(torch.full((10,), 7), 10).float(), digits).sum() == 1
    )
    assert confidence(certain).min() > 0.99
    assert entropy(certain).max() < 1e-3
    assert entropy(uniform) == pytest.approx(torch.full((10,), math.log(10)))


def test_predicted_class_entropy_detects_a_single_digit_for_everything() -> None:
    assert predicted_class_entropy(torch.arange(10), 10) == pytest.approx(1.0)
    assert predicted_class_entropy(torch.full((10,), 7), 10) == pytest.approx(0.0)


def test_mahalanobis_grows_with_distance_from_the_reference() -> None:
    torch.manual_seed(0)
    mean, precision = fit_gaussian(torch.randn(2048, 4))

    near = mahalanobis(torch.zeros(6, 4), mean, precision)
    far = mahalanobis(torch.full((6, 4), 8.0), mean, precision)
    assert (far > near).all()
    assert near.max() < 2.0


def test_spread_is_zero_for_identical_samples_and_positive_otherwise() -> None:
    torch.manual_seed(0)
    assert spread(torch.ones(8, *IMAGE)).item() == pytest.approx(0.0)
    assert spread(torch.randn(8, *IMAGE)).item() > 0.0


def test_label_accuracy_is_perfect_when_the_scores_identify_the_label() -> None:
    labels = all_combinations()
    log_scores = torch.full((180, 180), -10.0)
    log_scores[torch.arange(180), combination_index(labels)] = 0.0

    assert joint_label_accuracy(log_scores, labels).min() == 1.0
    for factor in range(3):
        assert factor_label_accuracy(log_scores, labels, factor).min() == 1.0


def test_label_accuracy_is_at_chance_when_the_scores_ignore_the_label() -> None:
    labels = all_combinations()
    log_scores = torch.zeros(180, 180)

    assert joint_label_accuracy(log_scores, labels).mean() < 0.05
    assert factor_label_accuracy(log_scores, labels, 0).mean() < 0.2

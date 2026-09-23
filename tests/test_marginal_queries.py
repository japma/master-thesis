"""Marginalized queries: the mixture reference and the calibration it is scored with."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from dataset_loaders.colour_mnist import NUM_FG
from evaluation.colour import BG_PALETTE, FG_PALETTE
from evaluation.conditionals import UNSPECIFIED, conditional, total_variation
from evaluation.marginal import (
    MIXTURE,
    as_query,
    calibration,
    calibration_histogram,
    free_factors,
    read_factor,
    run_marginal,
    sample_labels,
)
from evaluation.samples import BG, FG
from utils.config import (
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    GeneratedModelConfig,
    GenerationConfig,
    MarginalConfig,
    PretrainedAutoencoderConfig,
)

IMAGE_SIZE = 8
DEVICE = torch.device("cpu")

RED, GREEN = 0, 1
WHITE, BLACK = 0, 1


def painted(labels: torch.Tensor) -> torch.Tensor:
    """A centre square of each row's foreground colour on its background colour."""
    inset = IMAGE_SIZE // 4
    images = (
        torch.tensor(BG_PALETTE[labels[:, BG]])
        .reshape(-1, 3, 1, 1)
        .expand(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        .clone()
    )
    images[:, :, inset:-inset, inset:-inset] = torch.tensor(
        FG_PALETTE[labels[:, FG]]
    ).reshape(-1, 3, 1, 1)
    return images


def skewed_labels() -> torch.Tensor:
    """Digit 0 is 50% red / 50% green on black; digit 1 is uniform over six colours."""
    rows = []
    for _ in range(500):
        rows += [[0, RED, BLACK], [0, GREEN, BLACK]]
    for fg in range(NUM_FG):
        rows += [[1, fg, WHITE]] * 100
    return torch.tensor(rows)


# --- queries ---
def test_a_query_must_specify_the_digit() -> None:
    with pytest.raises(ValueError, match="digit must be specified"):
        as_query([-1, RED, BLACK])


def test_a_query_must_leave_something_to_the_model() -> None:
    with pytest.raises(ValueError, match="nothing is marginalized"):
        as_query([0, RED, BLACK])


def test_free_factors_are_the_unspecified_ones() -> None:
    assert free_factors(as_query([0, RED, -1])) == [BG]
    assert free_factors(as_query([0, -1, -1])) == [FG, BG]


# --- the mixture reference ---
def test_free_factors_are_drawn_from_the_training_conditional() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])

    drawn = sample_labels(query, 4000, labels, torch.Generator().manual_seed(0))

    assert (drawn[:, 0] == 0).all() and (drawn[:, BG] == BLACK).all()
    share_red = float((drawn[:, FG] == RED).float().mean())
    assert 0.45 < share_red < 0.55


def test_free_factors_are_drawn_jointly_not_independently() -> None:
    """fg and bg are perfectly correlated here, so independent draws would break it."""
    labels = torch.tensor([[0, RED, BLACK]] * 500 + [[0, GREEN, WHITE]] * 500)
    query = as_query([0, -1, -1])

    drawn = sample_labels(query, 2000, labels, torch.Generator().manual_seed(0))

    red = drawn[:, FG] == RED
    assert bool((drawn[red, BG] == BLACK).all())
    assert bool((drawn[~red, BG] == WHITE).all())


def test_a_query_with_no_training_rows_is_refused() -> None:
    """Digit 0 is only ever on black here, so a white background has nothing to draw."""
    labels = skewed_labels()
    with pytest.raises(ValueError, match="no training rows match"):
        sample_labels(as_query([0, -1, WHITE]), 10, labels)


def test_a_held_out_query_has_no_conditional_to_score_against() -> None:
    labels = skewed_labels()
    with pytest.raises(ValueError, match="held out"):
        conditional(labels, torch.tensor([0, UNSPECIFIED, WHITE]), FG)


# --- calibration ---
def test_a_correctly_distributed_set_scores_near_zero() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    drawn = sample_labels(query, 2000, labels, torch.Generator().manual_seed(0))

    scores = calibration(painted(drawn), query, labels)

    assert list(scores.columns) == ["digit", "fg", "bg", "factor", "value", "n"]
    assert scores["factor"].tolist() == ["fg"]
    assert scores["value"].max() < 0.05


def test_a_collapsed_set_is_caught() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    always_red = torch.tensor([[0, RED, BLACK]] * 2000)

    scores = calibration(painted(always_red), query, labels)

    # Half the training mass sits on green, and this set never emits it.
    assert scores["value"].iloc[0] == pytest.approx(0.5, abs=0.02)


def test_the_histogram_shows_both_distributions() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    always_red = torch.tensor([[0, RED, BLACK]] * 100)

    rows = calibration_histogram(painted(always_red), query, labels)

    assert len(rows) == NUM_FG
    assert rows["generated"].sum() == pytest.approx(1.0)
    assert rows["truth"].sum() == pytest.approx(1.0)
    assert rows.loc[rows["colour"] == RED, "generated"].item() == 1.0


def test_colours_are_read_back_off_the_pixels() -> None:
    labels = torch.tensor([[0, GREEN, BLACK], [1, RED, WHITE]])
    images = painted(labels)

    assert read_factor(images, FG).tolist() == [GREEN, RED]
    assert read_factor(images, BG).tolist() == [BLACK, WHITE]


def test_total_variation_is_zero_for_identical_distributions() -> None:
    p = torch.tensor([0.5, 0.3, 0.2])
    assert total_variation(p, p) == 0.0
    assert total_variation(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 1.0


# --- end to end ---
class PaintingSampler:
    """A 'model' whose latent is the label, so the decoder paints what was asked."""

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        return labels.float()


class PaintingDecoder:
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return painted(z.long())


def build_config(root: Path, queries: list[list[int]]) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        type="evaluation",
        dataset=DatasetConfig(
            name="colour_mnist_skewed",
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_classes=10,
        ),
        model=GeneratedModelConfig(name="cspn", model_type="cspn"),
        autoencoder=PretrainedAutoencoderConfig(name="vae", external=False),
        classifier=CheckpointConfig(name="judge"),
        generation=GenerationConfig(seed=0, batch_size=64),
        evaluation=EvaluationConfig(results_root=root / "results"),
        marginal=MarginalConfig(queries=queries, n_per_query=500),
    )


def patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluation.marginal.load_generative_model",
        lambda *args, **kwargs: (PaintingSampler(), "cspn:v2", Path("model.pt")),
    )
    monkeypatch.setattr(
        "evaluation.marginal.resolve_autoencoder",
        lambda *args, **kwargs: ("vae", "latest", False),
    )
    monkeypatch.setattr(
        "evaluation.marginal.load_vae",
        lambda *args, **kwargs: (PaintingDecoder(), "vae:v1"),
    )
    monkeypatch.setattr("evaluation.marginal.check_latent_dim", lambda *a, **k: None)
    monkeypatch.setattr(
        "evaluation.marginal.training_labels", lambda dataset: skewed_labels()
    )


def test_run_marginal_writes_both_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK]])
    patch_loaders(monkeypatch)

    run_marginal(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert sorted(p.name for p in results.glob("*.csv")) == [
        "colour_calibration.csv",
        "colour_calibration_histogram.csv",
    ]

    scores = pd.read_csv(results / "colour_calibration.csv")
    assert {"checkpoint", "seed", "std_correction", "source"} <= set(scores.columns)
    assert scores["source"].unique().tolist() == [MIXTURE]
    # The stub paints exactly what the mixture drew, so it sits at the floor.
    assert scores["value"].max() < 0.05


def test_re_running_replaces_rows_rather_than_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK]])
    patch_loaders(monkeypatch)
    path = cfg.evaluation.results_root / "colour_calibration.csv"

    run_marginal(cfg, DEVICE)
    once = pd.read_csv(path)
    run_marginal(cfg, DEVICE)
    twice = pd.read_csv(path)

    assert len(once) == len(twice)


def test_an_unknown_query_shape_fails_at_config_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="digit must be specified"):
        build_config(tmp_path, queries=[[-1, -1, BLACK]])

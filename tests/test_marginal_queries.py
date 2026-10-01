"""Marginalized queries: the mixture reference and the calibration it is scored with."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from dataset_loaders.colour_mnist import NUM_BG, NUM_FG
from evaluation.conditionals import UNSPECIFIED, conditional, total_variation
from evaluation.marginal import (
    DONT_CARE,
    MARGINALIZED,
    MIXTURE,
    REAL,
    answer,
    as_query,
    calibration,
    calibration_histogram,
    evaluate_marginal,
    free_factors,
    generate_marginal,
    model_arms,
    sample_labels,
)
from evaluation.pools import (
    MANIFEST_FILENAME,
    model_marginal_dir,
    real_marginal_dir,
)
from evaluation.samples import BG, DIGIT, FG
from models.cspn.joint_pc import JointPC
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import (
    CheckpointConfig,
    CSPNConfig,
    CSPNEncoderConfig,
    CSPNEncoderType,
    CSPNType,
    DatasetConfig,
    EvaluationConfig,
    JointPCConfig,
    MarginalConfig,
    PoolGenerationConfig,
    PoolModelConfig,
    PoolRunConfig,
)
from utils.reproducibility import seed_everything

IMAGE_SIZE = 8
DEVICE = torch.device("cpu")

RED, GREEN = 0, 1
WHITE, BLACK = 0, 1


def skewed_labels() -> torch.Tensor:
    """Digit 0 is 50% red / 50% green on black; digit 1 is uniform over six colours."""
    rows = []
    for _ in range(500):
        rows += [[0, RED, BLACK], [0, GREEN, BLACK]]
    for fg in range(NUM_FG):
        rows += [[1, fg, WHITE]] * 100
    return torch.tensor(rows)


# --- queries ---
def test_the_digit_can_be_left_free() -> None:
    assert free_factors(as_query([-1, RED, BLACK])) == [DIGIT]


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
        conditional(labels, torch.tensor([0, UNSPECIFIED, WHITE]), [FG])


# --- calibration ---
# The judge is taken to be perfect here: its predictions are the labels themselves.
def test_a_correctly_distributed_set_scores_near_zero() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    drawn = sample_labels(query, 2000, labels, torch.Generator().manual_seed(0))

    scores = calibration(drawn, query, labels)

    assert list(scores.columns) == ["digit", "fg", "bg", "factor", "value", "n"]
    assert scores["factor"].tolist() == ["fg"]
    assert scores["value"].max() < 0.05


def test_a_collapsed_set_is_caught() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    always_red = torch.tensor([[0, RED, BLACK]] * 2000)

    scores = calibration(always_red, query, labels)

    # Half the training mass sits on green, and this set never emits it.
    assert scores["value"].iloc[0] == pytest.approx(0.5, abs=0.02)


def test_several_free_factors_are_also_scored_jointly() -> None:
    labels = skewed_labels()
    drawn = sample_labels(as_query([-1, -1, -1]), 10, labels)

    scores = calibration(drawn, as_query([-1, -1, -1]), labels)

    assert scores["factor"].tolist() == ["digit", "fg", "bg", "digit+fg+bg"]


def test_the_joint_catches_independently_sampled_factors() -> None:
    """Right marginals, wrong dependence: red and black always go together in
    training, but this set pairs them at random."""
    labels = torch.tensor([[0, RED, BLACK]] * 500 + [[0, GREEN, WHITE]] * 500)
    query = as_query([0, -1, -1])
    generator = torch.Generator().manual_seed(0)
    independent = torch.stack(
        [
            torch.zeros(4000, dtype=torch.long),
            torch.tensor([RED, GREEN])[torch.randint(2, (4000,), generator=generator)],
            torch.tensor([BLACK, WHITE])[
                torch.randint(2, (4000,), generator=generator)
            ],
        ],
        dim=1,
    )

    scores = calibration(independent, query, labels).set_index("factor")["value"]

    assert scores["fg"] < 0.05 and scores["bg"] < 0.05
    assert scores["fg+bg"] == pytest.approx(0.5, abs=0.05)


def test_the_histogram_shows_both_distributions() -> None:
    labels = skewed_labels()
    query = as_query([0, -1, BLACK])
    always_red = torch.tensor([[0, RED, BLACK]] * 100)

    rows = calibration_histogram(always_red, query, labels)

    assert len(rows) == NUM_FG
    assert rows["generated"].sum() == pytest.approx(1.0)
    assert rows["truth"].sum() == pytest.approx(1.0)
    assert rows.loc[rows["class"] == RED, "generated"].item() == 1.0


def test_the_joint_histogram_is_laid_out_row_major() -> None:
    labels = torch.tensor([[0, GREEN, BLACK]] * 10)
    rows = calibration_histogram(labels, as_query([0, -1, -1]), labels)

    joint = rows[rows["factor"] == "fg+bg"]
    assert len(joint) == NUM_FG * NUM_BG
    assert joint.loc[joint["truth"] == 1.0, "class"].item() == GREEN * NUM_BG + BLACK


def test_total_variation_is_zero_for_identical_distributions() -> None:
    p = torch.tensor([0.5, 0.3, 0.2])
    assert total_variation(p, p) == 0.0
    assert total_variation(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 1.0


# --- the arms ---
class ZeroDecoder:
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)


def tiny_cspn(dropout: float) -> PsiNetCSPN:
    seed_everything(0)
    return PsiNetCSPN(
        config=CSPNConfig(
            model_type=CSPNType.PSINET,
            num_vars=4,
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
            min_var=1e-3,
            max_var=4.0,
            h_dims=[16],
            encoder_config=CSPNEncoderConfig(
                encoder_type=CSPNEncoderType.MULTI_CATEGORICAL,
                num_classes=[10, 6, 3],
                label_dropout_prob=dropout,
            ),
        )
    ).eval()


def tiny_joint_pc() -> JointPC:
    seed_everything(0)
    return JointPC(
        config=JointPCConfig(
            num_latents=4,
            label_cardinalities=[10, 6, 3],
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
        )
    ).eval()


def test_each_model_gets_the_arms_it_supports() -> None:
    assert model_arms(LabelSampler()) == [MIXTURE]
    assert model_arms(tiny_cspn(0.0)) == [MIXTURE]
    assert model_arms(tiny_cspn(0.3)) == [MIXTURE, DONT_CARE]
    assert model_arms(tiny_joint_pc()) == [MIXTURE, MARGINALIZED]


def test_dont_care_asks_with_the_unknown_index_for_free_factors() -> None:
    model = tiny_cspn(0.3)
    asked = []
    sample = model.sample
    model.sample = lambda labels, std_correction=1.0: (
        asked.append(labels),
        sample(labels, std_correction),
    )[1]
    query = as_query([3, -1, -1])

    latents, images, labels = answer(
        DONT_CARE,
        model,
        ZeroDecoder(),
        query,
        5,
        skewed_labels(),
        DEVICE,
        1.0,
        4,
        torch.Generator().manual_seed(0),
    )

    assert latents.shape == (5, 4) and images.shape == (5, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.cat(asked).tolist() == [[3, NUM_FG, NUM_BG]] * 5
    assert labels.tolist() == [[3, -1, -1]] * 5


def test_marginalized_keeps_the_given_factors_and_completes_the_rest() -> None:
    query = as_query([-1, GREEN, -1])

    _, images, labels = answer(
        MARGINALIZED,
        tiny_joint_pc(),
        ZeroDecoder(),
        query,
        6,
        skewed_labels(),
        DEVICE,
        1.0,
        4,
        torch.Generator().manual_seed(0),
    )

    assert images.shape[0] == 6
    assert (labels[:, FG] == GREEN).all()
    assert (labels >= 0).all()


# --- end to end, through the pool ---
class LabelSampler:
    """A 'model' whose latent is the label."""

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        return labels.float()


class LabelDecoder:
    """Writes the label into the pixels, one factor per channel."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 3, 1, 1) / 255.0


class LabelJudge:
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return (images * 255).round().long().flatten(1)


def build_config(
    root: Path,
    queries: list[list[int]],
    n: int = 500,
    models: list[PoolModelConfig] | None = None,
) -> PoolRunConfig:
    return PoolRunConfig(
        type="pools",
        dataset=DatasetConfig(
            name="colour_mnist_skewed",
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_classes=10,
        ),
        models=models or [PoolModelConfig(type="cspn", name="cspn")],
        classifier=CheckpointConfig(name="judge"),
        generation=PoolGenerationConfig(
            labels="stratified", batch_size=64, root=root / "pools"
        ),
        evaluation=EvaluationConfig(results_root=root / "results"),
        marginal=MarginalConfig(queries=queries, n_per_query=n),
    )


def patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = skewed_labels()
    monkeypatch.setattr(
        "evaluation.marginal.resolve_artifact",
        lambda name, tag="latest": f"{name}:v2",
    )
    monkeypatch.setattr(
        "evaluation.marginal.load_pool_model",
        lambda entry, ref, *a: (LabelSampler(), ref, LabelDecoder(), "vae:v1"),
    )
    monkeypatch.setattr("evaluation.marginal.check_latent_dim", lambda *a, **k: None)
    monkeypatch.setattr("evaluation.marginal.training_labels", lambda dataset: labels)
    # The training split, its images carrying their own labels like LabelDecoder's.
    monkeypatch.setattr(
        "evaluation.marginal.build_dataset",
        lambda *a, **k: TensorDataset(
            labels.float().reshape(-1, 3, 1, 1) / 255, labels
        ),
    )
    monkeypatch.setattr(
        "evaluation.marginal.load_judge", lambda *a, **k: (LabelJudge(), "judge:v3")
    )


def test_generate_then_evaluate_scores_the_real_set_and_every_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK], [0, -1, -1]])
    patch_loaders(monkeypatch)

    generate_marginal(cfg, DEVICE)
    evaluate_marginal(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert sorted(p.name for p in results.glob("*.csv")) == [
        "colour_calibration.csv",
        "colour_calibration_histogram.csv",
    ]
    scores = pd.read_csv(results / "colour_calibration.csv")
    assert sorted(scores["source"].unique()) == [MIXTURE, REAL]
    assert scores["classifier"].unique().tolist() == ["judge:v3"]
    assert scores.loc[scores["source"] == MIXTURE, "checkpoint"].unique().tolist() == [
        "cspn:v2"
    ]
    assert scores.loc[scores["source"] == REAL, "checkpoint"].isna().all()
    # One row per free factor, plus the joint one for the query with two free.
    assert len(scores) == 2 * (1 + 3)
    # Both render exactly what was drawn from training, so both sit at the floor.
    assert scores["value"].max() < 0.05


def test_a_second_run_skips_complete_sets_and_a_larger_n_resamples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_loaders(monkeypatch)
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK]], n=100)
    generate_marginal(cfg, DEVICE)
    real = real_marginal_dir(cfg.dataset_dir, [0, -1, BLACK]) / MANIFEST_FILENAME
    model = (
        model_marginal_dir(
            cfg.dataset_dir, 0, "cspn", "cspn:v2", 1.0, MIXTURE, [0, -1, BLACK]
        )
        / MANIFEST_FILENAME
    )
    written = (real.stat().st_mtime_ns, model.stat().st_mtime_ns)

    generate_marginal(cfg, DEVICE)
    assert (real.stat().st_mtime_ns, model.stat().st_mtime_ns) == written

    generate_marginal(build_config(tmp_path, queries=[[0, -1, BLACK]], n=200), DEVICE)
    assert real.stat().st_mtime_ns != written[0]
    assert model.stat().st_mtime_ns != written[1]


def test_re_scoring_replaces_rows_rather_than_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK], [1, -1, -1]])
    patch_loaders(monkeypatch)
    generate_marginal(cfg, DEVICE)
    path = cfg.evaluation.results_root / "colour_calibration.csv"

    evaluate_marginal(cfg, DEVICE)
    once = pd.read_csv(path)
    evaluate_marginal(cfg, DEVICE)
    twice = pd.read_csv(path)

    assert len(once) == len(twice)


def test_a_label_subset_model_is_not_queried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [
        PoolModelConfig(type="cspn", name="cspn"),
        PoolModelConfig(type="joint_pc", name="digit_only", labels=["digit"]),
        PoolModelConfig(type="gmm", name="gmm"),
    ]
    cfg = build_config(tmp_path, queries=[[0, -1, BLACK]], models=models)
    patch_loaders(monkeypatch)

    generate_marginal(cfg, DEVICE)

    sampled = sorted(p.name for p in (cfg.dataset_dir / "marginal" / "seed0").iterdir())
    assert sampled == ["cspn"]


def test_evaluating_before_generating_says_to_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_loaders(monkeypatch)
    with pytest.raises(FileNotFoundError, match="generate_pools"):
        evaluate_marginal(build_config(tmp_path, queries=[[0, -1, BLACK]]), DEVICE)


def test_an_unknown_query_shape_fails_at_config_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"a query is \[digit, fg, bg\]"):
        build_config(tmp_path, queries=[[-1, BLACK]])

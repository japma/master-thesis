"""The two evaluation stages against the contract they share: the pool directory."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError

from dataset_loaders.colour_mnist import (
    NUM_BG,
    NUM_COMBINATIONS,
    NUM_DIGITS,
    NUM_FG,
    all_combinations,
    combination_index,
)
from evaluation.colour import BG_PALETTE, FG_PALETTE
from evaluation.evaluate import GENERATED, evaluate_pool, predict, write_metric
from evaluation.generate import (
    MODEL_TYPES,
    check_latent_dim,
    empirical_labels,
    generate_pool,
    resolve_autoencoder,
    sample_and_decode,
    stratified_labels,
)
from evaluation.metrics import (
    METRICS,
    colour_accuracy,
    colour_accuracy_by_combination,
    colour_contrast,
    colour_drift,
    confusion_digit,
    digit_accuracy,
    digit_accuracy_by_combination,
    selected,
)
from evaluation.samples import (
    ReferenceManifest,
    SampleManifest,
    load_images,
    load_labels,
    load_latents,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
    save_pool,
    to_float,
    to_uint8,
)
from models.classifier import DigitClassifier
from utils.config import (
    CheckpointConfig,
    ClassifierConfig,
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    GeneratedModelConfig,
    GenerationConfig,
    PretrainedAutoencoderConfig,
)
from utils.config.loading import _apply_dataset_defaults
from utils.reproducibility import seed_everything

IMAGE_SIZE = 8
LATENT_DIM = 4
DEVICE = torch.device("cpu")


class ConstantSampler:
    """Latent is the digit, repeated -- so a decoded image encodes its own label."""

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        return labels[:, :1].float().expand(-1, LATENT_DIM).contiguous()


class StripeDecoder:
    """Stands in for the VAE: latent value -> a uniform brightness."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        brightness = (z[:, :1] / NUM_DIGITS).clamp(0, 1)
        return brightness.view(-1, 1, 1, 1).expand(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).expand(-1, LATENT_DIM)


def build_judge() -> DigitClassifier:
    seed_everything(0)
    return DigitClassifier(
        config=ClassifierConfig(image_size=IMAGE_SIZE, conv_channels=[4, 4, 8])
    ).eval()


def write_pool(
    root: Path,
    n_per_cell: int = 1,
    vae_checkpoint: str = "vae:v1",
    pool_dir: Path | None = None,
) -> Path:
    """A complete pool -- samples plus the nested reference pool -- built by hand."""
    labels = stratified_labels(n_per_cell)
    n = labels.shape[0]
    pool_dir = pool_dir if pool_dir is not None else root / "pool"
    save_pool(
        pool_dir,
        SampleManifest(
            model_checkpoint="cspn:v2",
            model_type="cspn",
            vae_checkpoint=vae_checkpoint,
            dataset="colour_mnist_uniform",
            schedule="stratified",
            n_per_cell=n_per_cell,
            n_samples=n,
            seed=0,
            std_correction=1.0,
            git_commit="abc123",
        ),
        latents=torch.zeros(n, LATENT_DIM),
        images=torch.zeros(n, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
        labels=labels,
    )

    reference_labels = all_combinations()
    m = reference_labels.shape[0]
    save_pool(
        reference_dir(pool_dir),
        ReferenceManifest(
            vae_checkpoint=vae_checkpoint,
            dataset="colour_mnist_uniform",
            split="val",
            n_samples=m,
            git_commit="abc123",
        ),
        latents=torch.zeros(m, LATENT_DIM),
        images=torch.zeros(m, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
        labels=reference_labels,
        originals=torch.zeros(m, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
    )
    return pool_dir


def build_config(
    root: Path,
    n_per_cell: int = 1,
    seed: int = 0,
    metrics: list[str] | None = None,
) -> EvaluationRunConfig:
    """A config whose `pool_dir` is where `write_pool` puts its pool."""
    return EvaluationRunConfig(
        type="evaluation",
        dataset=DatasetConfig(
            name="colour_mnist_uniform",
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_classes=10,
        ),
        model=GeneratedModelConfig(name="cspn", model_type="cspn"),
        autoencoder=PretrainedAutoencoderConfig(name="vae", external=False),
        classifier=CheckpointConfig(name="judge"),
        generation=GenerationConfig(n_per_cell=n_per_cell, seed=seed, output_root=root),
        evaluation=EvaluationConfig(
            results_root=root / "results", batch_size=64, metrics=metrics
        ),
    )


# --- schedule ---
def test_stratified_schedule_covers_every_combination_equally() -> None:
    labels = stratified_labels(3)

    assert labels.shape == (NUM_COMBINATIONS * 3, 3)
    assert combination_index(labels).bincount().unique().tolist() == [3]


def test_stratified_schedule_rejects_an_empty_cell() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        stratified_labels(0)


# --- generation ---
def test_sample_and_decode_keeps_one_image_per_label_in_order() -> None:
    labels = stratified_labels(2)

    latents, images = sample_and_decode(
        ConstantSampler(), StripeDecoder(), labels, DEVICE, batch_size=64
    )

    assert latents.shape == (labels.shape[0], LATENT_DIM)
    assert images.shape == (labels.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.dtype == torch.uint8
    # The fake sampler puts the digit in the latent, so order is checkable.
    assert torch.equal(latents[:, 0], labels[:, 0].float())


def test_uint8_round_trip_stays_within_a_quantisation_step() -> None:
    images = torch.rand(16, 3, IMAGE_SIZE, IMAGE_SIZE)

    assert torch.allclose(to_float(to_uint8(images)), images, atol=1 / 255)


# --- pool contract ---
def test_pool_round_trips_through_disk(tmp_path: Path) -> None:
    pool_dir = write_pool(tmp_path, n_per_cell=2)

    manifest = load_sample_manifest(pool_dir)
    assert manifest.model_checkpoint == "cspn:v2"
    assert manifest.n_samples == NUM_COMBINATIONS * 2
    assert load_labels(pool_dir).shape == (NUM_COMBINATIONS * 2, 3)
    assert load_latents(pool_dir).shape == (NUM_COMBINATIONS * 2, LATENT_DIM)
    assert load_images(pool_dir).dtype == torch.uint8

    reference = load_reference_manifest(reference_dir(pool_dir))
    assert reference.split == "val"
    assert load_originals(reference_dir(pool_dir)).shape[0] == NUM_COMBINATIONS


def test_reading_a_reference_pool_as_a_sample_pool_is_refused(tmp_path: Path) -> None:
    pool_dir = write_pool(tmp_path)

    with pytest.raises(ValueError, match="not a 'samples' one"):
        load_sample_manifest(reference_dir(pool_dir))


def test_save_pool_rejects_tensors_that_disagree(tmp_path: Path) -> None:
    manifest = ReferenceManifest(
        vae_checkpoint="vae:v1",
        dataset="colour_mnist_uniform",
        split="val",
        n_samples=4,
        git_commit=None,
    )
    with pytest.raises(ValueError, match="disagree on length"):
        save_pool(
            tmp_path / "bad",
            manifest,
            latents=torch.zeros(4, LATENT_DIM),
            images=torch.zeros(3, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
            labels=all_combinations()[:4],
        )


def test_save_pool_rejects_float_images(tmp_path: Path) -> None:
    manifest = ReferenceManifest(
        vae_checkpoint="vae:v1",
        dataset="colour_mnist_uniform",
        split="val",
        n_samples=4,
        git_commit=None,
    )
    with pytest.raises(ValueError, match="must be uint8"):
        save_pool(
            tmp_path / "bad",
            manifest,
            latents=torch.zeros(4, LATENT_DIM),
            images=torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE),
            labels=all_combinations()[:4],
        )


# --- generation guards ---
def test_generation_checks_the_latent_dim_before_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A decoder that cannot read the model's latents must fail naming both artifacts,
    not as a matmul error thousands of samples later."""
    cfg = build_config(tmp_path, n_per_cell=1)
    monkeypatch.setattr(
        "evaluation.generate.load_generative_model",
        lambda *a, **k: (WideSampler(LATENT_DIM + 4), "cspn:v9", Path("m.pt")),
    )
    monkeypatch.setattr(
        "evaluation.generate.resolve_autoencoder",
        lambda *a, **k: ("vae", "latest", False),
    )
    monkeypatch.setattr(
        "evaluation.generate.load_vae", lambda *a, **k: (NarrowVAE(), "vae:v1")
    )

    with pytest.raises(ValueError, match="were not trained together"):
        generate_pool(cfg, DEVICE)


# --- label schedules ---
def test_empirical_schedule_draws_whole_rows_from_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attributes are dependent, so a drawn row must be a row that actually occurred."""
    training = torch.tensor([[0, 1, 1], [1, 0, 0]] * 50)
    monkeypatch.setattr("evaluation.generate.training_labels", lambda name: training)

    drawn = empirical_labels("whatever", 200, torch.Generator().manual_seed(0))

    assert drawn.shape == (200, 3)
    rows = {tuple(row) for row in drawn.tolist()}
    assert rows <= {(0, 1, 1), (1, 0, 0)}
    assert len(rows) == 2


def test_empirical_schedule_rejects_an_empty_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evaluation.generate.training_labels", lambda name: torch.zeros(4, 3).long()
    )
    with pytest.raises(ValueError, match="at least 1"):
        empirical_labels("whatever", 0)


def test_a_pool_accepts_a_label_space_that_is_not_colour_mnists(tmp_path: Path) -> None:
    """CelebA carries 40 binary attributes, not [digit, fg, bg]."""
    manifest = ReferenceManifest(
        vae_checkpoint="vae:v1",
        dataset="celeba",
        split="val",
        n_samples=4,
        git_commit=None,
    )
    save_pool(
        tmp_path / "celeba",
        manifest,
        latents=torch.zeros(4, LATENT_DIM),
        images=torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
        labels=torch.randint(0, 2, (4, 40)),
    )
    assert load_labels(tmp_path / "celeba").shape == (4, 40)


def test_a_judged_metric_without_a_judge_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2)
    cfg.classifier = None
    write_pool(tmp_path, n_per_cell=2, pool_dir=cfg.pool_dir)

    with pytest.raises(ValueError, match="evaluate_sets"):
        evaluate_pool(cfg, DEVICE)


# --- metrics ---
def test_every_metric_names_the_csv_it_writes() -> None:
    assert [metric.FILENAME for metric in METRICS] == [
        "digit_accuracy.csv",
        "digit_accuracy_by_combination.csv",
        "confusion_digit.csv",
        "colour_accuracy.csv",
        "colour_accuracy_by_combination.csv",
        "colour_drift.csv",
        "colour_contrast.csv",
    ]


def blank(labels: torch.Tensor) -> torch.Tensor:
    """Flat images, for the metrics that never look at pixels."""
    return torch.zeros(labels.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)


def unused(labels: torch.Tensor) -> torch.Tensor:
    """Judge predictions, for the colour metrics that never look at them."""
    return torch.zeros(labels.shape[0], dtype=torch.long)


def painted(labels: torch.Tensor) -> torch.Tensor:
    """A centre square of each row's foreground colour on its background colour.

    The square clears `BORDER_MARGIN`, so the border reads pure background and the most
    foreground-like decile reads pure foreground.
    """
    inset = IMAGE_SIZE // 4
    images = (
        torch.tensor(BG_PALETTE[labels[:, 2]])
        .reshape(-1, 3, 1, 1)
        .expand(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        .clone()
    )
    images[:, :, inset:-inset, inset:-inset] = torch.tensor(
        FG_PALETTE[labels[:, 1]]
    ).reshape(-1, 3, 1, 1)
    return images


def test_accuracy_counts_matches() -> None:
    labels = torch.tensor([[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]])
    predictions = torch.tensor([1, 2, 3, 4])

    scores = digit_accuracy.compute(blank(labels), predictions, labels, num_classes=10)

    assert list(scores.columns) == ["value", "n"]
    assert scores["value"].tolist() == [0.5]
    assert scores["n"].tolist() == [4]


def test_accuracy_by_combination_has_a_row_per_cell_present() -> None:
    labels = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 2, 1]])
    predictions = torch.tensor([0, 9, 1])

    cells = digit_accuracy_by_combination.compute(
        blank(labels), predictions, labels, num_classes=10
    )

    assert list(cells.columns) == ["digit", "fg", "bg", "value", "n"]
    assert len(cells) == 2
    assert cells["value"].tolist() == [0.5, 1.0]
    assert cells["n"].tolist() == [2, 1]


def test_accuracy_by_combination_covers_every_cell_of_a_stratified_pool() -> None:
    labels = stratified_labels(2)
    predictions = torch.zeros(labels.shape[0], dtype=torch.long)

    cells = digit_accuracy_by_combination.compute(
        blank(labels), predictions, labels, num_classes=10
    )

    assert len(cells) == NUM_COMBINATIONS
    assert int(cells["n"].sum()) == NUM_COMBINATIONS * 2
    assert set(cells["digit"]) == set(range(NUM_DIGITS))
    assert set(cells["fg"]) == set(range(NUM_FG))
    assert set(cells["bg"]) == set(range(NUM_BG))


def test_confusion_is_a_complete_grid_with_truth_as_rows() -> None:
    labels = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    predictions = torch.tensor([0, 1, 1])

    pairs = confusion_digit.compute(blank(labels), predictions, labels, num_classes=2)

    assert list(pairs.columns) == ["truth", "predicted", "n"]
    assert len(pairs) == 4
    assert int(pairs["n"].sum()) == 3
    lookup = pairs.set_index(["truth", "predicted"])["n"]
    assert lookup[(0, 0)] == 1 and lookup[(0, 1)] == 1
    assert lookup[(1, 0)] == 0 and lookup[(1, 1)] == 1


def test_colour_is_read_perfectly_off_correctly_painted_images() -> None:
    labels = all_combinations()
    images = painted(labels)

    scores = colour_accuracy.compute(images, unused(labels), labels, num_classes=10)
    drift = colour_drift.compute(images, unused(labels), labels, num_classes=10)

    assert scores["factor"].tolist() == ["fg", "bg"]
    assert scores["value"].tolist() == [1.0, 1.0]
    assert drift["value"].max() < 1e-5


def test_colour_accuracy_catches_a_swapped_background() -> None:
    labels = all_combinations()
    wrong = labels.clone()
    wrong[:, 2] = (wrong[:, 2] + 1) % NUM_BG

    scores = colour_accuracy.compute(
        painted(wrong), unused(labels), labels, num_classes=10
    )
    drift = colour_drift.compute(painted(wrong), unused(labels), labels, num_classes=10)

    background = scores[scores["factor"] == "bg"].iloc[0]
    assert background["value"] == 0.0
    assert drift[drift["factor"] == "bg"].iloc[0]["value"] > 0.1


def test_colour_accuracy_by_combination_has_a_row_per_cell_and_factor() -> None:
    labels = all_combinations()

    cells = colour_accuracy_by_combination.compute(
        painted(labels), unused(labels), labels, num_classes=10
    )

    assert list(cells.columns) == ["factor", "digit", "fg", "bg", "value", "n"]
    assert len(cells) == 2 * NUM_COMBINATIONS
    assert cells["value"].min() == 1.0


def test_contrast_separates_a_painted_digit_from_a_flat_image() -> None:
    labels = all_combinations()

    painted_contrast = colour_contrast.compute(
        painted(labels), unused(labels), labels, num_classes=10
    )
    flat_contrast = colour_contrast.compute(
        blank(labels), unused(labels), labels, num_classes=10
    )

    assert float(flat_contrast["value"].iloc[0]) == 0.0
    assert float(painted_contrast["value"].iloc[0]) > 0.5


# --- evaluation ---
def test_predict_returns_one_digit_per_image() -> None:
    images = torch.randint(0, 255, (20, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)

    predictions = predict(build_judge(), images, DEVICE, batch_size=8)

    assert predictions.shape == (20,)
    assert predictions.min() >= 0 and predictions.max() < 10


def patch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluation.evaluate.load_judge",
        lambda cfg, device: (build_judge(), "judge:v7"),
    )


def test_evaluate_pool_writes_one_csv_per_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2)
    write_pool(tmp_path, n_per_cell=2, pool_dir=cfg.pool_dir)
    patch_judge(monkeypatch)

    evaluate_pool(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert sorted(p.name for p in results.glob("*.csv")) == [
        "colour_accuracy.csv",
        "colour_accuracy_by_combination.csv",
        "colour_contrast.csv",
        "colour_drift.csv",
        "confusion_digit.csv",
        "digit_accuracy.csv",
        "digit_accuracy_by_combination.csv",
    ]


def test_every_csv_identifies_the_run_and_the_image_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2)
    write_pool(tmp_path, n_per_cell=2, pool_dir=cfg.pool_dir)
    patch_judge(monkeypatch)

    evaluate_pool(cfg, DEVICE)

    results = cfg.evaluation.results_root

    for path in results.glob("*.csv"):
        frame = pd.read_csv(path)
        assert {"checkpoint", "seed", "std_correction", "source"} <= set(frame.columns)
        assert frame["checkpoint"].unique().tolist() == ["cspn:v2"]
        assert frame["classifier"].unique().tolist() == ["judge:v7"]
        assert set(frame["source"]) == {"generated", "real", "reconstruction"}

    scores = pd.read_csv(results / "digit_accuracy.csv")
    generated = scores[scores["source"] == GENERATED].iloc[0]
    assert int(generated["n"]) == NUM_COMBINATIONS * 2

    cells = pd.read_csv(results / "digit_accuracy_by_combination.csv")
    assert len(cells) == 3 * NUM_COMBINATIONS


def test_the_config_selects_which_metrics_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2, metrics=["colour_accuracy"])
    write_pool(tmp_path, n_per_cell=2, pool_dir=cfg.pool_dir)
    patch_judge(monkeypatch)

    evaluate_pool(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert [p.name for p in results.glob("*.csv")] == ["colour_accuracy.csv"]


def test_omitting_metrics_runs_all_of_them() -> None:
    assert selected(None) == METRICS


def test_an_unknown_metric_fails_at_config_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown metrics"):
        build_config(tmp_path, metrics=["colour_acuracy"])


def test_an_empty_metric_list_fails_rather_than_writing_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="omit the key"):
        build_config(tmp_path, metrics=[])


def test_a_repeated_metric_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="twice"):
        build_config(tmp_path, metrics=["colour_drift", "colour_drift"])


def test_re_evaluating_a_run_replaces_its_rows_rather_than_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2)
    write_pool(tmp_path, n_per_cell=2, pool_dir=cfg.pool_dir)
    patch_judge(monkeypatch)
    results = cfg.evaluation.results_root

    evaluate_pool(cfg, DEVICE)
    once = pd.read_csv(results / "digit_accuracy.csv")
    evaluate_pool(cfg, DEVICE)
    twice = pd.read_csv(results / "digit_accuracy.csv")

    assert len(once) == len(twice) == 3


def test_a_second_run_appends_alongside_the_first(tmp_path: Path) -> None:
    path = tmp_path / "digit_accuracy.csv"
    first = pd.DataFrame(
        [{"checkpoint": "a:v1", "seed": 0, "std_correction": 1.0, "value": 0.5}]
    )
    second = pd.DataFrame(
        [{"checkpoint": "b:v1", "seed": 0, "std_correction": 1.0, "value": 0.9}]
    )

    write_metric(path, first, {"checkpoint": "a:v1", "seed": 0, "std_correction": 1.0})
    write_metric(path, second, {"checkpoint": "b:v1", "seed": 0, "std_correction": 1.0})

    frame = pd.read_csv(path)
    assert frame["checkpoint"].tolist() == ["a:v1", "b:v1"]


def test_evaluate_pool_refuses_a_ceiling_from_a_different_vae(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path)
    write_pool(tmp_path, pool_dir=cfg.pool_dir)
    reference = reference_dir(cfg.pool_dir) / "manifest.json"
    data = json.loads(reference.read_text())
    data["vae_checkpoint"] = "other_vae:v9"
    reference.write_text(json.dumps(data))
    patch_judge(monkeypatch)

    with pytest.raises(ValueError, match="would not bound the samples"):
        evaluate_pool(cfg, DEVICE)


# --- config ---
def test_pool_dir_is_determined_by_the_config_alone() -> None:
    cfg = build_config(Path("results/samples"), seed=3)

    assert cfg.pool_dir == Path("results/samples/colour_mnist_uniform__cspn__seed3")


def test_changing_the_seed_moves_the_pool(tmp_path: Path) -> None:
    assert (
        build_config(tmp_path, seed=0).pool_dir
        != build_config(tmp_path, seed=1).pool_dir
    )


def test_shipped_configs_validate() -> None:
    for path in sorted(Path("configs/evaluation").glob("*.yaml")):
        raw = _apply_dataset_defaults(yaml.safe_load(path.read_text()))
        cfg = EvaluationRunConfig.model_validate(raw)
        assert cfg.type == "evaluation"
        assert cfg.dataset.channels == 3
        assert cfg.model.model_type in MODEL_TYPES


def test_an_unknown_config_key_is_refused(tmp_path: Path) -> None:
    raw = _apply_dataset_defaults(
        yaml.safe_load(
            (Path("configs/evaluation") / "colour_mnist_uniform.yaml").read_text()
        )
    )
    raw["generation"]["n_per_cel"] = 10  # typo

    with pytest.raises(ValidationError, match="n_per_cel"):
        EvaluationRunConfig.model_validate(raw)


def test_evaluate_pool_says_which_pool_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path)
    patch_judge(monkeypatch)

    with pytest.raises(FileNotFoundError, match="No sample pool at"):
        evaluate_pool(cfg, DEVICE)


# --- pairing the model with the decoder it was trained against ---
class WideSampler:
    """Samples wider latents than the decoder under test can read."""

    def __init__(self, width: int) -> None:
        self.width = width

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        return torch.zeros(labels.shape[0], self.width)


class NarrowVAE(torch.nn.Module):
    def get_latent_dim(self) -> torch.Size:
        return torch.Size([LATENT_DIM])


def test_check_latent_dim_names_both_artifacts_on_a_mismatch() -> None:
    with pytest.raises(ValueError, match="were not trained together"):
        check_latent_dim(
            WideSampler(LATENT_DIM + 4),
            NarrowVAE(),
            DEVICE,
            "psinet_colour_mnist_uniform:v9",
            "variational_colour_mnist_uniform:v1",
            all_combinations()[:1],
        )


def test_check_latent_dim_passes_a_matching_pair() -> None:
    check_latent_dim(
        WideSampler(LATENT_DIM),
        NarrowVAE(),
        DEVICE,
        "model:v1",
        "vae:v1",
        all_combinations()[:1],
    )


def test_check_latent_dim_probes_with_the_runs_own_label_space() -> None:
    """CelebA hands it a 40-attribute row; nothing here may assume three factors."""
    check_latent_dim(
        WideSampler(LATENT_DIM),
        NarrowVAE(),
        DEVICE,
        "model:v1",
        "vae:v1",
        torch.randint(0, 2, (4, 40)),
    )


def test_a_pinned_autoencoder_wins_over_the_recorded_one(tmp_path: Path) -> None:
    pinned = PretrainedAutoencoderConfig(name="pinned_vae", external=False, tag="v2")

    assert resolve_autoencoder(pinned, tmp_path / "nope.pt", "model:v1") == (
        "pinned_vae",
        "v2",
        False,
    )


def test_an_unpinned_autoencoder_comes_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "evaluation.generate.read_source_artifact", lambda path: "recorded_vae:v7"
    )

    assert resolve_autoencoder(None, tmp_path / "model.pt", "model:v1") == (
        "recorded_vae:v7",
        "latest",
        False,
    )


def test_an_unresolvable_autoencoder_says_to_pin_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evaluation.generate.read_source_artifact", lambda path: None)
    monkeypatch.setattr("evaluation.generate.trained_with", lambda ref: None)

    with pytest.raises(ValueError, match="Pin one in the config"):
        resolve_autoencoder(None, tmp_path / "model.pt", "model:v1")


# --- set metrics ---
def test_set_metrics_write_one_row_per_halving_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.sets as sets

    def stub(device: torch.device):
        generator = torch.Generator().manual_seed(0)
        return lambda images: (
            images.float().flatten(1)[:, :6]
            + torch.randn(images.shape[0], 6, generator=generator)
        )

    monkeypatch.setattr(sets, "NETWORKS", dict.fromkeys(sets.NETWORKS, stub))
    cfg = build_config(tmp_path, n_per_cell=1)
    cfg.evaluation.set_metrics = list(sets.SET_METRICS)
    cfg.evaluation.halvings = 2
    write_pool(tmp_path, n_per_cell=1, pool_dir=cfg.pool_dir)

    sets.run_sets(cfg, DEVICE)

    for name in sets.SET_METRICS:
        table = pd.read_csv(tmp_path / "results" / f"{name}.csv")
        assert len(table) == 2 * 3
        assert set(table["source"]) == {"real", "reconstruction", "generated"}
        assert (table["n"] == 90).all()


def test_an_unknown_set_metric_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown set metrics"):
        EvaluationConfig(results_root=tmp_path, set_metrics=["fid", "isc"])

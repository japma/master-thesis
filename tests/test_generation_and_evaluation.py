"""The two evaluation stages against the contract they share: a dataset's pool."""

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import TensorDataset

from dataset_loaders.colour_mnist import (
    NUM_BG,
    NUM_COMBINATIONS,
    NUM_DIGITS,
    NUM_FG,
    all_combinations,
    combination_index,
)
from evaluation.colour import BG_PALETTE, FG_PALETTE
from evaluation.evaluate import (
    GENERATED,
    REAL,
    RECONSTRUCTION,
    evaluate_pools,
    find_models,
    predict,
    write_metric,
)
from evaluation.generate import (
    MODEL_TYPES,
    check_latent_dim,
    generate_pools,
    resolve_autoencoder,
    sample_and_decode,
    stratified_labels,
)
from evaluation.metrics import accuracy, accuracy_by_combination, confusion
from evaluation.pools import (
    IMAGES,
    LABELS,
    LATENTS,
    ModelManifest,
    RealManifest,
    is_complete,
    load_conditioning,
    load_manifest,
    load_tensor,
    model_dir,
    real_dir,
    vae_dir,
    write_dir,
)
from evaluation.samples import to_float, to_uint8
from models.classifier import DigitClassifier
from utils.config import (
    CheckpointConfig,
    ClassifierConfig,
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    PoolGenerationConfig,
    PoolModelConfig,
    PoolRunConfig,
    PretrainedAutoencoderConfig,
)
from utils.config.loading import _apply_dataset_defaults
from utils.reproducibility import seed_everything

IMAGE_SIZE = 8
LATENT_DIM = 4
DEVICE = torch.device("cpu")
DATASET = "colour_mnist_uniform"


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

    def get_latent_dim(self) -> torch.Size:
        return torch.Size([LATENT_DIM])


def build_judge() -> DigitClassifier:
    seed_everything(0)
    return DigitClassifier(
        config=ClassifierConfig(image_size=IMAGE_SIZE, conv_channels=[4, 4, 8])
    ).eval()


def real_split(n_per_cell: int = 1) -> TensorDataset:
    """A stand-in val split: every colour-MNIST cell, painted."""
    labels = all_combinations().repeat_interleave(n_per_cell, dim=0)
    return TensorDataset(painted(labels), labels)


def build_config(
    root: Path,
    labels: str = "stratified",
    n_per_cell: int = 1,
    models: list[PoolModelConfig] | None = None,
    seeds: list[int] | None = None,
    metrics: list[str] | None = None,
) -> PoolRunConfig:
    return PoolRunConfig(
        type="pools",
        dataset=DatasetConfig(
            name=DATASET,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_classes=10,
        ),
        models=models or [PoolModelConfig(type="cspn", name="cspn")],
        classifier=CheckpointConfig(name="judge"),
        generation=PoolGenerationConfig(
            labels=labels,
            n_per_cell=n_per_cell,
            seeds=seeds or [0],
            batch_size=64,
            root=root / "pools",
        ),
        evaluation=EvaluationConfig(
            results_root=root / "results", batch_size=64, metrics=metrics
        ),
    )


def patch_generation(
    monkeypatch: pytest.MonkeyPatch,
    versions: dict[str, int] | None = None,
    sampler: object | None = None,
    n_real_per_cell: int = 1,
) -> dict[str, int]:
    """Stub wandb, the checkpoints and the val split. Returns the live version table:
    a name absent from it is a model wandb does not have."""
    versions = {"cspn": 2} if versions is None else versions

    def resolve(name: str, tag: str = "latest") -> str | None:
        if name not in versions:
            return None
        return f"{name}:{tag}" if tag != "latest" else f"{name}:v{versions[name]}"

    monkeypatch.setattr("evaluation.generate.resolve_artifact", resolve)
    monkeypatch.setattr(
        "evaluation.generate.load_generative_model",
        lambda model_type, ref, device: (
            sampler or ConstantSampler(),
            ref,
            Path("m.pt"),
        ),
    )
    monkeypatch.setattr(
        "evaluation.generate.resolve_autoencoder",
        lambda cfg, path, ref: ("vae:v1", "latest", False),
    )
    monkeypatch.setattr(
        "evaluation.generate.load_vae", lambda *a, **k: (StripeDecoder(), "vae:v1")
    )
    monkeypatch.setattr(
        "evaluation.generate.build_dataset",
        lambda *a, **k: real_split(n_real_per_cell),
    )
    monkeypatch.setattr("evaluation.generate.LOADER_WORKERS", 0)
    return versions


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
def test_a_directory_is_written_all_or_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "real"
    crashed = tmp_path / "real.partial"
    crashed.mkdir()
    (crashed / "images.pt").write_bytes(b"half a file")

    assert not is_complete(directory)
    manifest = RealManifest(dataset=DATASET, split="val", n=4, git_commit=None)
    write_dir(
        directory,
        manifest,
        {
            IMAGES: torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
            LABELS: all_combinations()[:4],
        },
    )

    assert is_complete(directory)
    assert not crashed.exists()
    assert load_manifest(directory, RealManifest) == manifest


def test_write_dir_rejects_tensors_that_disagree(tmp_path: Path) -> None:
    manifest = RealManifest(dataset=DATASET, split="val", n=4, git_commit=None)
    with pytest.raises(ValueError, match="disagree"):
        write_dir(
            tmp_path / "bad",
            manifest,
            {
                IMAGES: torch.zeros(3, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
                LABELS: all_combinations()[:4],
            },
        )


def test_write_dir_rejects_float_images(tmp_path: Path) -> None:
    manifest = RealManifest(dataset=DATASET, split="val", n=4, git_commit=None)
    with pytest.raises(ValueError, match="must be uint8"):
        write_dir(
            tmp_path / "bad",
            manifest,
            {IMAGES: torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE)},
        )


def test_reading_one_kind_of_directory_as_another_is_refused(tmp_path: Path) -> None:
    write_dir(
        tmp_path / "real",
        RealManifest(dataset=DATASET, split="val", n=4, git_commit=None),
        {LABELS: all_combinations()[:4]},
    )

    with pytest.raises(ValueError, match="not 'samples'"):
        load_manifest(tmp_path / "real", ModelManifest)


def test_a_label_space_that_is_not_colour_mnists_fits(tmp_path: Path) -> None:
    """CelebA's 40 binary attributes."""
    write_dir(
        tmp_path / "real",
        RealManifest(dataset="celeba", split="val", n=4, git_commit=None),
        {LABELS: torch.randint(0, 2, (4, 40))},
    )
    assert load_tensor(tmp_path / "real", LABELS).shape == (4, 40)


# --- generate_pools ---
def test_generate_pools_writes_real_vae_and_model_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, n_per_cell=2)
    patch_generation(monkeypatch)

    report = generate_pools(cfg, DEVICE)

    root = cfg.dataset_dir
    assert load_tensor(real_dir(root), LABELS).shape == (NUM_COMBINATIONS, 3)
    assert load_conditioning(root, "stratified_2").shape == (NUM_COMBINATIONS * 2, 3)
    assert load_tensor(vae_dir(root, "vae:v1"), IMAGES).shape[0] == NUM_COMBINATIONS
    samples = model_dir(root, 0, "cspn", "cspn:v2", 1.0)
    manifest = load_manifest(samples, ModelManifest)
    assert manifest.model_checkpoint == "cspn:v2"
    assert manifest.vae_checkpoint == "vae:v1"
    assert manifest.labels == "stratified_2"
    assert load_tensor(samples, LATENTS).shape == (NUM_COMBINATIONS * 2, LATENT_DIM)
    assert load_tensor(samples, IMAGES).dtype == torch.uint8
    assert len(report.generated) == 1


def test_a_second_run_skips_what_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path)
    patch_generation(monkeypatch)

    generate_pools(cfg, DEVICE)
    again = generate_pools(cfg, DEVICE)

    assert again.generated == []
    assert len(again.skipped) == 1


def test_a_new_version_is_sampled_beside_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path)
    versions = patch_generation(monkeypatch)
    generate_pools(cfg, DEVICE)

    versions["cspn"] = 3
    report = generate_pools(cfg, DEVICE)

    assert len(report.generated) == 1
    assert is_complete(model_dir(cfg.dataset_dir, 0, "cspn", "cspn:v2", 1.0))
    assert is_complete(model_dir(cfg.dataset_dir, 0, "cspn", "cspn:v3", 1.0))


def test_a_model_wandb_does_not_have_is_reported_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    models = [
        PoolModelConfig(type="cspn", name="cspn"),
        PoolModelConfig(type="joint_pc", name="joint_pc_anchored"),
    ]
    cfg = build_config(tmp_path, models=models)
    patch_generation(monkeypatch)

    report = generate_pools(cfg, DEVICE)

    assert report.missing == ["joint_pc_anchored:latest"]
    assert len(report.generated) == 1
    assert "WARNING: joint_pc_anchored:latest" in capsys.readouterr().out


def test_real_labels_condition_sample_i_on_image_i(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, labels="real")
    patch_generation(monkeypatch, n_real_per_cell=2)

    generate_pools(cfg, DEVICE)

    samples = model_dir(cfg.dataset_dir, 0, "cspn", "cspn:v2", 1.0)
    real_labels = load_tensor(real_dir(cfg.dataset_dir), LABELS)
    # The stub puts the digit in the latent, so the pairing is checkable.
    assert torch.equal(load_tensor(samples, LATENTS)[:, 0], real_labels[:, 0].float())
    assert not (cfg.dataset_dir / "stratified_1").exists()


def test_a_label_subset_model_sees_only_its_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[torch.Tensor] = []

    class Recording(ConstantSampler):
        def sample(self, labels: torch.Tensor, std_correction: float = 1.0):
            seen.append(labels)
            return super().sample(labels, std_correction)

    models = [PoolModelConfig(type="joint_pc", name="digit_only", labels=("digit",))]
    cfg = build_config(tmp_path, models=models)
    patch_generation(monkeypatch, versions={"digit_only": 1}, sampler=Recording())

    generate_pools(cfg, DEVICE)

    assert all(labels.shape[1] == 1 for labels in seen)
    samples = model_dir(cfg.dataset_dir, 0, "joint_pc", "digit_only:v1", 1.0)
    assert load_manifest(samples, ModelManifest).label_columns == [0]


def test_every_seed_gets_its_own_samples_and_shares_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path, seeds=[0, 1])
    patch_generation(monkeypatch)

    generate_pools(cfg, DEVICE)

    for seed in (0, 1):
        assert is_complete(model_dir(cfg.dataset_dir, seed, "cspn", "cspn:v2", 1.0))
    assert sorted(p.name for p in cfg.dataset_dir.iterdir()) == [
        "real",
        "seed0",
        "seed1",
        "stratified_1",
        "vaes",
    ]


def test_generation_checks_the_latent_dim_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoder that cannot read the model's latents must fail naming both artifacts,
    not as a matmul error thousands of samples later."""
    cfg = build_config(tmp_path)
    patch_generation(monkeypatch, sampler=WideSampler(LATENT_DIM + 4))

    with pytest.raises(ValueError, match="were not trained together"):
        generate_pools(cfg, DEVICE)


# --- metrics ---
FACTORS = ["digit", "fg", "bg"]
CARDINALITIES = [NUM_DIGITS, NUM_FG, NUM_BG]


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


def test_accuracy_scores_every_factor() -> None:
    labels = torch.tensor([[1, 0, 2], [2, 3, 0], [0, 0, 0], [0, 5, 1]])
    predictions = torch.tensor([[1, 0, 1], [2, 3, 1], [3, 1, 0], [4, 5, 0]])

    scores = accuracy(predictions, labels, FACTORS, CARDINALITIES)

    assert list(scores.columns) == ["factor", "value", "n"]
    assert scores["factor"].tolist() == FACTORS
    assert scores["value"].tolist() == [0.5, 0.75, 0.25]
    assert scores["n"].tolist() == [4, 4, 4]


def test_accuracy_scores_only_the_factors_it_is_given() -> None:
    scores = accuracy(
        torch.tensor([[1], [3]]), torch.tensor([[1], [2]]), ["digit"], [NUM_DIGITS]
    )

    assert scores["factor"].tolist() == ["digit"]
    assert scores["value"].tolist() == [0.5]


def test_accuracy_by_combination_has_a_row_per_factor_and_cell_present() -> None:
    labels = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 2, 1]])
    predictions = torch.tensor([[0, 0, 1], [9, 0, 1], [1, 3, 1]])

    cells = accuracy_by_combination(predictions, labels, FACTORS, CARDINALITIES)

    assert list(cells.columns) == ["factor", "digit", "fg", "bg", "value", "n"]
    assert len(cells) == 3 * 2
    digit = cells[cells["factor"] == "digit"]
    assert digit["value"].tolist() == [0.5, 1.0]
    assert digit["n"].tolist() == [2, 1]
    assert cells[cells["factor"] == "fg"]["value"].tolist() == [1.0, 0.0]
    assert cells[cells["factor"] == "bg"]["value"].tolist() == [0.0, 1.0]


def test_accuracy_by_combination_covers_every_cell_of_a_stratified_pool() -> None:
    labels = stratified_labels(2)

    cells = accuracy_by_combination(labels.clone(), labels, FACTORS, CARDINALITIES)

    assert len(cells) == 3 * NUM_COMBINATIONS
    assert int(cells["n"].sum()) == 3 * NUM_COMBINATIONS * 2
    assert cells["value"].min() == 1.0
    assert set(cells["digit"]) == set(range(NUM_DIGITS))
    assert set(cells["fg"]) == set(range(NUM_FG))
    assert set(cells["bg"]) == set(range(NUM_BG))


def test_a_digit_only_set_is_grouped_by_digit_alone() -> None:
    labels = torch.tensor([[0], [0], [3]])

    cells = accuracy_by_combination(
        torch.tensor([[0], [1], [3]]), labels, ["digit"], [NUM_DIGITS]
    )

    assert list(cells.columns) == ["factor", "digit", "value", "n"]
    assert cells["value"].tolist() == [0.5, 1.0]


def test_confusion_is_a_complete_grid_per_factor_with_truth_as_rows() -> None:
    labels = torch.tensor([[0, 0], [0, 0], [1, 2]])
    predictions = torch.tensor([[0, 0], [1, 2], [1, 2]])

    pairs = confusion(predictions, labels, ["digit", "bg"], [2, NUM_BG])

    assert list(pairs.columns) == ["factor", "truth", "predicted", "n"]
    digit = pairs[pairs["factor"] == "digit"].set_index(["truth", "predicted"])["n"]
    assert len(digit) == 4 and int(digit.sum()) == 3
    assert digit[(0, 0)] == 1 and digit[(0, 1)] == 1
    assert digit[(1, 0)] == 0 and digit[(1, 1)] == 1
    bg = pairs[pairs["factor"] == "bg"].set_index(["truth", "predicted"])["n"]
    assert len(bg) == NUM_BG**2 and int(bg.sum()) == 3
    assert bg[(0, 0)] == 1 and bg[(0, 2)] == 1 and bg[(2, 2)] == 1


# --- evaluation ---
def test_predict_returns_one_class_per_factor_and_image() -> None:
    images = torch.randint(0, 255, (20, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)

    predictions = predict(build_judge(), images, DEVICE, batch_size=8)

    assert predictions.shape == (20, 3)
    assert predictions.min() >= 0
    assert (
        predictions.max(dim=0).values < torch.tensor([NUM_DIGITS, NUM_FG, NUM_BG])
    ).all()


def patch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evaluation.evaluate.load_judge",
        lambda cfg, device: (build_judge(), "judge:v7"),
    )


def generated_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config
) -> PoolRunConfig:
    cfg = build_config(tmp_path, n_per_cell=2, **config)
    patch_generation(monkeypatch)
    generate_pools(cfg, DEVICE)
    return cfg


def test_evaluate_pools_writes_one_csv_per_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = generated_pool(tmp_path, monkeypatch)
    patch_judge(monkeypatch)

    evaluate_pools(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert sorted(p.name for p in results.glob("*.csv")) == [
        "accuracy.csv",
        "accuracy_by_combination.csv",
        "confusion.csv",
    ]


def test_every_csv_identifies_the_set_and_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = generated_pool(tmp_path, monkeypatch)
    patch_judge(monkeypatch)

    evaluate_pools(cfg, DEVICE)

    results = cfg.evaluation.results_root
    for path in results.glob("*.csv"):
        frame = pd.read_csv(path)
        assert {"checkpoint", "seed", "std_correction", "vae", "source"} <= set(
            frame.columns
        )
        assert frame["classifier"].unique().tolist() == ["judge:v7"]
        by_source = dict(list(frame.groupby("source")))
        assert set(by_source) == {GENERATED, REAL, RECONSTRUCTION}
        assert by_source[GENERATED]["checkpoint"].unique().tolist() == ["cspn:v2"]
        assert by_source[GENERATED]["vae"].unique().tolist() == ["vae:v1"]
        assert by_source[RECONSTRUCTION]["vae"].unique().tolist() == ["vae:v1"]
        assert by_source[RECONSTRUCTION]["checkpoint"].isna().all()
        assert by_source[REAL][["checkpoint", "seed", "vae"]].isna().all().all()

    scores = pd.read_csv(results / "accuracy.csv")
    generated = scores[scores["source"] == GENERATED]
    assert generated["factor"].tolist() == FACTORS
    assert generated["n"].unique().tolist() == [NUM_COMBINATIONS * 2]

    cells = pd.read_csv(results / "accuracy_by_combination.csv")
    assert len(cells) == 3 * len(FACTORS) * NUM_COMBINATIONS


def test_the_real_and_reconstruction_sets_are_scored_once_for_every_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [
        PoolModelConfig(type="cspn", name="cspn"),
        PoolModelConfig(type="cspn", name="cspn", std_correction=0.5),
    ]
    cfg = generated_pool(tmp_path, monkeypatch, models=models)
    patch_judge(monkeypatch)

    evaluate_pools(cfg, DEVICE)

    scores = pd.read_csv(cfg.evaluation.results_root / "accuracy.csv")
    counts = scores.groupby("source").size()
    assert counts[REAL] == counts[RECONSTRUCTION] == len(FACTORS)
    assert counts[GENERATED] == 2 * len(FACTORS)
    assert sorted(scores["std_correction"].dropna().unique()) == [0.5, 1.0]


def test_a_model_is_scored_only_on_the_factors_it_was_conditioned_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [
        PoolModelConfig(type="cspn", name="cspn"),
        PoolModelConfig(type="cspn", name="cspn", labels=["digit"], std_correction=0.5),
    ]
    cfg = generated_pool(tmp_path, monkeypatch, models=models)
    patch_judge(monkeypatch)

    evaluate_pools(cfg, DEVICE)

    results = cfg.evaluation.results_root
    scores = pd.read_csv(results / "accuracy.csv")
    digit_only = scores[scores["std_correction"] == 0.5]
    assert digit_only["factor"].tolist() == ["digit"]
    cells = pd.read_csv(results / "accuracy_by_combination.csv")
    assert len(cells[cells["std_correction"] == 0.5]) == NUM_DIGITS


def test_the_config_selects_which_metrics_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = generated_pool(tmp_path, monkeypatch, metrics=["accuracy"])
    patch_judge(monkeypatch)

    evaluate_pools(cfg, DEVICE)

    results = cfg.evaluation.results_root
    assert [p.name for p in results.glob("*.csv")] == ["accuracy.csv"]


def test_an_unknown_metric_fails_at_config_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown metrics"):
        build_config(tmp_path, metrics=["acuracy"])


def test_an_empty_metric_list_fails_rather_than_writing_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="omit the key"):
        build_config(tmp_path, metrics=[])


def test_a_repeated_metric_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="twice"):
        build_config(tmp_path, metrics=["accuracy", "accuracy"])


def test_re_scoring_a_set_replaces_its_rows_rather_than_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = generated_pool(tmp_path, monkeypatch)
    patch_judge(monkeypatch)
    results = cfg.evaluation.results_root

    evaluate_pools(cfg, DEVICE)
    once = pd.read_csv(results / "accuracy.csv")
    evaluate_pools(cfg, DEVICE)
    twice = pd.read_csv(results / "accuracy.csv")

    assert len(once) == len(twice) == 3 * len(FACTORS)


def test_a_second_run_appends_alongside_the_first(tmp_path: Path) -> None:
    path = tmp_path / "accuracy.csv"
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


def test_a_judged_metric_without_a_judge_says_so(tmp_path: Path) -> None:
    cfg = build_config(tmp_path)
    cfg.classifier = None

    with pytest.raises(ValueError, match="evaluate_sets"):
        evaluate_pools(cfg, DEVICE)


def test_evaluation_takes_the_newest_version_unless_one_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = build_config(tmp_path)
    versions = patch_generation(monkeypatch)
    generate_pools(cfg, DEVICE)
    versions["cspn"] = 10
    generate_pools(cfg, DEVICE)

    [(_, newest)] = find_models(cfg)
    cfg.models[0].version = "v2"
    [(_, pinned)] = find_models(cfg)

    assert newest.model_checkpoint == "cspn:v10"
    assert pinned.model_checkpoint == "cspn:v2"


def test_a_listed_model_with_nothing_generated_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    cfg = generated_pool(tmp_path, monkeypatch)
    cfg.models.append(PoolModelConfig(type="joint_pc", name="never_trained"))

    assert len(find_models(cfg)) == 1
    assert "nothing generated for never_trained" in capsys.readouterr().out


def test_an_empty_pool_says_to_generate_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_pools"):
        find_models(build_config(tmp_path))


# --- config ---
def test_shipped_pool_configs_validate() -> None:
    paths = sorted(Path("configs/pools").glob("*.yaml"))
    assert paths
    for path in paths:
        raw = _apply_dataset_defaults(yaml.safe_load(path.read_text()))
        cfg = PoolRunConfig.model_validate(raw)
        assert cfg.dataset.channels == 3
        assert all(model.type in MODEL_TYPES for model in cfg.models)


def test_shipped_marginal_configs_validate() -> None:
    for path in sorted(Path("configs/evaluation").glob("*.yaml")):
        raw = _apply_dataset_defaults(yaml.safe_load(path.read_text()))
        cfg = EvaluationRunConfig.model_validate(raw)
        assert cfg.marginal is not None


def test_an_unknown_config_key_is_refused() -> None:
    raw = _apply_dataset_defaults(
        yaml.safe_load(Path("configs/pools/colour_mnist_uniform.yaml").read_text())
    )
    raw["generation"]["n_per_cel"] = 10  # typo

    with pytest.raises(ValidationError, match="n_per_cel"):
        PoolRunConfig.model_validate(raw)


def test_a_version_must_be_a_version() -> None:
    with pytest.raises(ValidationError, match="look like v3"):
        PoolModelConfig(type="cspn", name="cspn", version="latest")


def test_a_model_listed_twice_is_refused(tmp_path: Path) -> None:
    twice = [PoolModelConfig(type="cspn", name="cspn")] * 2
    with pytest.raises(ValidationError, match="twice"):
        build_config(tmp_path, models=twice)


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
def stub_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.sets as sets

    def stub(device: torch.device):
        generator = torch.Generator().manual_seed(0)
        return lambda images: (
            images.float().flatten(1)[:, :6]
            + torch.randn(images.shape[0], 6, generator=generator)
        )

    monkeypatch.setattr(sets, "NETWORKS", dict.fromkeys(sets.NETWORKS, stub))


def test_set_metrics_write_one_row_per_halving_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.sets as sets

    stub_networks(monkeypatch)
    cfg = generated_pool(tmp_path, monkeypatch)
    cfg.evaluation.set_metrics = list(sets.SET_METRICS)
    cfg.evaluation.halvings = 2

    sets.run_sets(cfg, DEVICE)

    for name in sets.SET_METRICS:
        table = pd.read_csv(tmp_path / "results" / f"{name}.csv")
        assert len(table) == 2 * 3
        assert set(table["source"]) == {"real", "reconstruction", "generated"}
        assert (table["n"] == NUM_COMBINATIONS // 2).all()


def test_paired_halvings_score_the_held_out_halfs_own_samples() -> None:
    from evaluation.sets import halvings

    n, splits = halvings(10, 10, 3, seed=0, paired=True)
    _, unpaired = halvings(10, 40, 3, seed=0, paired=False)

    assert n == 5
    for (reference, held_out, generated), (ref_u, held_u, _) in zip(
        splits, unpaired, strict=True
    ):
        assert torch.equal(generated, held_out)
        assert set(reference.tolist()).isdisjoint(held_out.tolist())
        # The real splits depend on the seed alone, so every model shares them.
        assert torch.equal(reference, ref_u) and torch.equal(held_out, held_u)


def test_an_unknown_set_metric_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown set metrics"):
        EvaluationConfig(results_root=tmp_path, set_metrics=["fid", "isc"])


def test_a_flat_label_dataset_hands_the_model_flat_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MNIST yields `(N,)` labels, and the categorical encoder expects exactly that."""
    seen: list[torch.Tensor] = []

    class Recording(ConstantSampler):
        def sample(self, labels: torch.Tensor, std_correction: float = 1.0):
            seen.append(labels)
            return labels.float().unsqueeze(1).expand(-1, LATENT_DIM).contiguous()

    cfg = build_config(tmp_path, labels="real")
    patch_generation(monkeypatch, sampler=Recording())
    digits = torch.arange(20) % NUM_DIGITS
    monkeypatch.setattr(
        "evaluation.generate.build_dataset",
        lambda *a, **k: TensorDataset(
            torch.rand(20, 3, IMAGE_SIZE, IMAGE_SIZE), digits
        ),
    )

    generate_pools(cfg, DEVICE)

    assert all(labels.dim() == 1 for labels in seen)
    assert load_tensor(real_dir(cfg.dataset_dir), LABELS).shape == (20, 1)

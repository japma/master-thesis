"""The eval harness and its metrics, checked against inputs whose answers are known.

Uses stand-in models rather than trained circuits: every metric is meant to be
independent of what produced the samples, so the tests are too.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation import (
    BG_PALETTE,
    FG_PALETTE,
    ColourFidelity,
    DigitAccuracy,
    LabelDiscrimination,
    LatentPlausibility,
    NegativeLogLikelihood,
    SampleDiversity,
    all_combinations,
    run_eval_suite,
)
from evaluation.classifier import DigitClassifier
from evaluation.harness import DensityBatch, SampleBatch

NUM_LATENTS = 4
NUM_COMBINATIONS = 180
IMAGE = (3, 28, 28)


def painted(labels: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
    """Images literally painted in each row's intended colours, so colour metrics have
    a known-correct answer: a centre square of foreground on a background field."""
    labels_np = labels.cpu().numpy()
    images = torch.tensor(
        BG_PALETTE[labels_np[:, 2]], dtype=torch.float32
    ).reshape(-1, 3, 1, 1).expand(-1, *IMAGE).clone()
    fg = torch.tensor(FG_PALETTE[labels_np[:, 1]], dtype=torch.float32)
    images[:, :, 10:18, 10:18] = fg.reshape(-1, 3, 1, 1)
    if noise:
        images = (images + noise * torch.randn_like(images)).clamp(0.0, 1.0)
    return images


class PaintingModel(nn.Module):
    """Stands in for a generative model: emits a latent carrying the label, which the
    decoder below turns back into a correctly-coloured image."""

    def __init__(self, jitter: float = 0.0) -> None:
        super().__init__()
        self.jitter = jitter
        self.dummy = nn.Parameter(torch.zeros(1))

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        latents = labels.float()
        noise = self.jitter * std_correction * torch.randn(
            labels.shape[0], NUM_LATENTS - labels.shape[1]
        )
        return torch.cat([latents, noise], dim=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # A density that is maximal when the latent's label part matches `labels`.
        return -((z[:, :3] - labels.float()) ** 2).sum(dim=1)


class PaintingDecoder(nn.Module):
    def __init__(self, jitter: float = 0.0) -> None:
        super().__init__()
        self.jitter = jitter

    def eval(self) -> "PaintingDecoder":
        return self

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # A real model's latents carry no label indices, so clamp into palette range:
        # the stub only has to produce *an* image, not the right one.
        indices = z[:, :3].round().long()
        for column, size in enumerate((NUM_DIGITS, NUM_FG, NUM_BG)):
            indices[:, column] = indices[:, column].clamp(0, size - 1)
        return painted(indices, noise=self.jitter)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def sample_batch(labels: torch.Tensor, jitter: float = 0.0) -> SampleBatch:
    return SampleBatch(
        images=painted(labels, noise=jitter),
        latents=labels.float(),
        labels=labels,
    )


def test_colour_fidelity_is_perfect_on_correctly_painted_images() -> None:
    metric = ColourFidelity()
    metric.update(sample_batch(all_combinations()))
    tables = metric.compute().tables

    assert np.nanmin(tables["bg_accuracy"]) == 1.0
    assert np.nanmin(tables["fg_accuracy"]) == 1.0
    assert np.nanmax(tables["bg_drift"]) < 1e-5
    assert np.nanmin(tables["contrast"]) > 0.0


def test_colour_fidelity_catches_a_swapped_palette() -> None:
    labels = all_combinations()
    wrong = labels.clone()
    wrong[:, 2] = (wrong[:, 2] + 1) % 3

    metric = ColourFidelity()
    metric.update(
        SampleBatch(images=painted(wrong), latents=labels.float(), labels=labels)
    )
    assert np.nanmax(metric.compute().tables["bg_accuracy"]) == 0.0


def test_diversity_is_zero_for_a_point_predictor_and_positive_otherwise() -> None:
    labels = all_combinations()[:8].repeat_interleave(4, dim=0)

    identical = SampleDiversity(samples_per_combination=4)
    identical.update(sample_batch(labels))
    tables = identical.compute().tables
    assert np.nanmax(tables["pixel_std"]) == pytest.approx(0.0, abs=1e-6)
    assert np.nanmax(tables["latent_std"]) == pytest.approx(0.0, abs=1e-6)

    torch.manual_seed(0)
    varied = SampleDiversity(samples_per_combination=4)
    varied.update(
        SampleBatch(
            images=painted(labels, noise=0.2),
            latents=labels.float() + torch.randn(labels.shape[0], 3),
            labels=labels,
        )
    )
    varied_tables = varied.compute().tables
    assert np.nanmin(varied_tables["pixel_std"]) > 0.0
    assert np.nanmin(varied_tables["latent_std"]) > 0.0


def test_diversity_rejects_partial_combinations() -> None:
    metric = SampleDiversity(samples_per_combination=4)
    with pytest.raises(ValueError, match="not whole combinations"):
        metric.update(sample_batch(all_combinations()[:3]))

    with pytest.raises(ValueError, match="at least 2"):
        SampleDiversity(samples_per_combination=1)


def test_digit_accuracy_reads_the_classifier() -> None:
    labels = all_combinations()

    class AlwaysCorrect(DigitClassifier):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.one_hot(labels[:, 0], 10).float() * 20.0

    class AlwaysSeven(DigitClassifier):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.one_hot(
                torch.full((images.shape[0],), 7), 10
            ).float() * 20.0

    correct = DigitAccuracy(AlwaysCorrect())
    correct.update(sample_batch(labels))
    result = correct.compute()
    assert np.nanmin(result.tables["accuracy"]) == 1.0
    # Ten digits predicted evenly -> maximum normalized entropy.
    assert result.scalars["predicted_digit_entropy"] == pytest.approx(1.0, abs=1e-6)

    stuck = DigitAccuracy(AlwaysSeven())
    stuck.update(sample_batch(labels))
    stuck_result = stuck.compute()
    assert stuck_result.tables["accuracy"][7].mean() == 1.0
    assert stuck_result.tables["accuracy"][0].mean() == 0.0
    # One digit for everything -> no entropy at all, the mode-collapse signature.
    assert stuck_result.scalars["predicted_digit_entropy"] == pytest.approx(0.0)


def test_latent_plausibility_grows_with_distance_from_the_reference() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(2048, NUM_LATENTS))
    metric = LatentPlausibility(reference)

    labels = all_combinations()[:6]
    near = torch.zeros(6, NUM_LATENTS)
    far = torch.full((6, NUM_LATENTS), 8.0)

    metric.update(SampleBatch(images=painted(labels), latents=near, labels=labels))
    close = np.nanmean(metric.compute().tables["mahalanobis"])

    metric = LatentPlausibility(reference)
    metric.update(SampleBatch(images=painted(labels), latents=far, labels=labels))
    distant = np.nanmean(metric.compute().tables["mahalanobis"])

    assert distant > close
    assert close < 2.0


def test_test_log_likelihood_reports_the_negative_log_prob() -> None:
    labels = all_combinations()[:6]
    metric = NegativeLogLikelihood()
    metric.update(
        DensityBatch(
            latents=torch.zeros(6, NUM_LATENTS),
            labels=labels,
            log_prob=torch.full((6,), -2.5),
            score=lambda z, y: torch.zeros(z.shape[0]),
        )
    )
    table = metric.compute().tables["value"]
    assert np.nanmax(table) == pytest.approx(2.5)


def test_label_discrimination_is_perfect_when_the_density_identifies_the_label() -> None:
    model = PaintingModel()
    labels = all_combinations()
    latents = torch.cat([labels.float(), torch.zeros(len(labels), 1)], dim=1)

    metric = LabelDiscrimination()
    metric.update(
        DensityBatch(
            latents=latents,
            labels=labels,
            log_prob=model(latents, labels),
            score=model.forward,
        )
    )
    tables = metric.compute().tables
    for key in ("joint_accuracy", "digit_accuracy", "fg_accuracy", "bg_accuracy"):
        assert np.nanmin(tables[key]) == 1.0


def test_label_discrimination_is_at_chance_for_an_unconditional_density() -> None:
    labels = all_combinations()
    metric = LabelDiscrimination()
    metric.update(
        DensityBatch(
            latents=torch.zeros(len(labels), NUM_LATENTS),
            labels=labels,
            log_prob=torch.zeros(len(labels)),
            # Ignores the label entirely, so no combination can be preferred.
            score=lambda z, y: torch.zeros(z.shape[0]),
        )
    )
    tables = metric.compute().tables
    assert np.nanmean(tables["joint_accuracy"]) < 0.05
    assert np.nanmean(tables["digit_accuracy"]) < 0.2


def test_suite_runs_both_families_and_splits_by_held_out() -> None:
    torch.manual_seed(0)
    model, decoder = PaintingModel(jitter=0.3), PaintingDecoder(jitter=0.05)

    seen = np.ones((10, 6, 3), dtype=bool)
    seen[0, 0, 0] = False
    seen[3, 2, 1] = False

    report = run_eval_suite(
        model,
        decoder,
        torch.device("cpu"),
        sample_metrics=[ColourFidelity(), SampleDiversity(4)],
        seen=seen,
        samples_per_combination=4,
    )

    assert report.counts.sum() == NUM_COMBINATIONS * 4
    assert set(report.tables) == {
        "colour/bg_accuracy",
        "colour/fg_accuracy",
        "colour/bg_drift",
        "colour/fg_drift",
        "colour/contrast",
        "diversity/pixel_std",
        "diversity/latent_std",
    }

    trained, held_out = report.split("colour/bg_accuracy")
    assert 0.0 <= held_out <= 1.0
    assert 0.0 <= trained <= 1.0

    summary = report.summary()
    assert "colour/bg_accuracy/held_out" in summary
    assert set(report.marginals("colour/bg_accuracy")) == {"digit", "fg", "bg"}


def test_suite_needs_a_loader_for_density_metrics() -> None:
    with pytest.raises(ValueError, match="need a `loader`"):
        run_eval_suite(
            PaintingModel(),
            PaintingDecoder(),
            torch.device("cpu"),
            density_metrics=[NegativeLogLikelihood()],
        )


def test_suite_accepts_a_real_circuit() -> None:
    """The harness must take an actual model, not just the stand-ins above."""
    from models.cspn.joint_pc import JointPC
    from utils.config import JointPCConfig
    from utils.reproducibility import seed_everything

    seed_everything(0)
    model = JointPC(
        config=JointPCConfig(
            num_latents=3,
            label_cardinalities=[10, 6, 3],
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
        )
    )

    report = run_eval_suite(
        model,
        PaintingDecoder(),
        torch.device("cpu"),
        sample_metrics=[ColourFidelity(), SampleDiversity(2)],
        samples_per_combination=2,
    )
    assert report.counts.sum() == NUM_COMBINATIONS * 2
    assert np.isfinite(report.overall("colour/bg_accuracy"))

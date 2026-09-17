"""The collection loops, run with stand-in models whose answers are known."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG, all_combinations
from evaluation.classifier import DigitClassifier
from evaluation.collect import (
    ImageSet,
    combination_table,
    encode_images,
    evaluate_model,
    mark_seen,
    sample_images,
    score_density,
    score_images,
    spread_by_combination,
)
from evaluation.colour import BG_PALETTE, FG_PALETTE
from evaluation.metrics import fit_gaussian

NUM_LATENTS = 4
IMAGE = (3, 28, 28)
DEVICE = torch.device("cpu")


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


class PaintingModel(nn.Module):
    """Samples a latent carrying its label; its density peaks where the latent matches."""

    def __init__(self, jitter: float = 0.0) -> None:
        super().__init__()
        self.jitter = jitter

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        noise = self.jitter * std_correction * torch.randn(labels.shape[0], 1)
        return torch.cat([labels.float(), noise], dim=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -((z[:, :3] - labels.float()) ** 2).sum(dim=1)


class PaintingAE(nn.Module):
    """Decodes a latent's label part into a correctly painted image."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        indices = z[:, :3].round().long()
        for column, size in enumerate((NUM_DIGITS, NUM_FG, NUM_BG)):
            indices[:, column] = indices[:, column].clamp(0, size - 1)
        return painted(indices)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], NUM_LATENTS)


def real_loader(copies: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    labels = all_combinations()
    return [(painted(labels), labels) for _ in range(copies)]


def gaussian() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return fit_gaussian(torch.randn(512, NUM_LATENTS))


def judge() -> DigitClassifier:
    torch.manual_seed(0)
    return DigitClassifier()


def test_sample_images_covers_every_combination_in_order() -> None:
    samples = sample_images(
        PaintingModel(), PaintingAE(), DEVICE, samples_per_combination=3
    )

    assert samples.labels.shape == (540, 3)
    assert torch.equal(samples.labels[::3], all_combinations())
    assert samples.images.shape == (540, *IMAGE)
    assert torch.equal(samples.latents[:, :3], samples.labels.float())


def test_score_images_gives_one_row_per_image() -> None:
    samples = sample_images(
        PaintingModel(), PaintingAE(), DEVICE, samples_per_combination=2
    )
    frame = score_images(samples, judge(), gaussian(), DEVICE, batch_size=100)

    assert len(frame) == 360
    assert frame["bg_accuracy"].min() == 1.0
    assert frame["fg_accuracy"].min() == 1.0
    assert (frame[["digit", "fg", "bg"]].to_numpy() == samples.labels.numpy()).all()
    assert frame["mahalanobis"].notna().all()


def test_spread_groups_by_label_not_by_position() -> None:
    """Real test images arrive in no particular order, so grouping must use the labels."""
    torch.manual_seed(0)
    labels = all_combinations().repeat(3, 1)
    order = torch.randperm(len(labels))
    images = painted(labels) + 0.1 * torch.randn(len(labels), *IMAGE)
    shuffled = ImageSet(labels[order], torch.zeros(len(labels), 4), images[order])

    frame = spread_by_combination(shuffled)
    assert len(frame) == 180
    assert frame["pixel_std"].min() > 0.0
    assert frame["latent_std"].max() == pytest.approx(0.0)


def test_score_density_reports_the_nll_and_identifies_the_label() -> None:
    model = PaintingModel()
    labels = all_combinations()
    real = ImageSet(labels, model.sample(labels), painted(labels))

    frame = score_density(model, real, DEVICE, max_images=100, batch_size=32)
    assert len(frame) == 100
    expected = -model(real.latents[:100], labels[:100]).numpy()
    np.testing.assert_allclose(frame["nll"], expected, atol=1e-6)
    for column in ("joint", "digit", "fg", "bg"):
        assert frame[f"{column}_label_accuracy"].min() == 1.0


def test_evaluate_model_scores_samples_next_to_real_data() -> None:
    seen = np.ones((NUM_DIGITS, NUM_FG, NUM_BG), dtype=bool)
    seen[1, :, 0] = False

    torch.manual_seed(0)
    evaluation = evaluate_model(
        PaintingModel(jitter=0.3),
        PaintingAE(),
        judge(),
        train_loader=real_loader(1),
        test_loader=real_loader(2),
        device=DEVICE,
        seen=seen,
        samples_per_combination=4,
        density_images=64,
    )

    counts = evaluation.images["source"].value_counts()
    assert counts["sample"] == 720
    assert counts["real"] == counts["reconstruction"] == 360
    assert set(evaluation.spread["source"]) == {"sample", "real"}
    assert len(evaluation.density) == 64
    assert (~evaluation.images["seen"]).sum() == 6 * (4 + 2 + 2)
    assert evaluation.samples.labels.shape == (720, 3)


def test_encode_images_keeps_labels_with_their_images() -> None:
    real = encode_images(PaintingAE(), real_loader(2), DEVICE)
    assert real.labels.shape == (360, 3)
    assert torch.equal(real.images, painted(real.labels))


def test_tables_and_seen_marks_follow_the_label_columns() -> None:
    samples = sample_images(
        PaintingModel(), PaintingAE(), DEVICE, samples_per_combination=2
    )
    frame = score_images(samples, judge(), gaussian(), DEVICE)

    table = combination_table(frame, "bg_accuracy")
    assert table.shape == (NUM_DIGITS, NUM_FG, NUM_BG)
    assert np.nanmin(table) == 1.0

    seen = np.ones((NUM_DIGITS, NUM_FG, NUM_BG), dtype=bool)
    seen[3, 2, 1] = False
    marked = mark_seen(frame, seen)
    assert set(
        marked.loc[~marked["seen"], ["digit", "fg", "bg"]].itertuples(index=False)
    ) == {(3, 2, 1)}


def test_a_real_circuit_runs_through_the_loops() -> None:
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

    samples = sample_images(model, PaintingAE(), DEVICE, samples_per_combination=2)
    assert samples.latents.shape == (360, 3)

    frame = score_density(model, samples, DEVICE, max_images=64)
    assert np.isfinite(frame["nll"]).all()

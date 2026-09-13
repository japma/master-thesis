"""Sources, and the pairing rules that keep a metric away from evidence it cannot read."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation import BG_PALETTE, FG_PALETTE, ColourFidelity, Pass, run_suite
from evaluation.batch import EvalBatch
from evaluation.harness import Metric, MetricResult
from evaluation.metrics import NegativeLogLikelihood, SampleDiversity
from evaluation.sources import (
    DensitySource,
    RealSource,
    ReconstructionSource,
    SampleSource,
    all_combinations,
)
from models.autoencoder import AbstractAutoencoder

NUM_LATENTS = 4
IMAGE = (3, 28, 28)
DEVICE = torch.device("cpu")

BLACK, GREEN = 1, 1  # indices into BG_PALETTE / FG_PALETTE


def field(colour: np.ndarray, count: int = 1) -> torch.Tensor:
    return (
        torch.tensor(colour, dtype=torch.float32)
        .reshape(1, 3, 1, 1)
        .expand(count, *IMAGE)
        .clone()
    )


class StubAE(AbstractAutoencoder):
    """Encodes to the image's mean colour and decodes back to a flat field of it."""

    def get_latent_dim(self) -> torch.Size:
        return torch.Size([NUM_LATENTS])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), torch.zeros(x.shape[0], 1)], dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :3].reshape(-1, 3, 1, 1).expand(-1, *IMAGE).clone()


class StubModel(nn.Module):
    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        return torch.zeros(labels.shape[0], NUM_LATENTS)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -labels[:, 0].float()


def one_batch_loader(count: int = 12) -> list:
    labels = all_combinations()[:count]
    return [(field(BG_PALETTE[BLACK], count), labels)]


def test_sample_source_yields_whole_combinations_and_counts_them() -> None:
    source = SampleSource(
        StubModel(), StubAE(), DEVICE, samples_per_combination=4,
        combinations_per_chunk=8,
    )
    sizes = [batch.size for batch in source.batches()]

    assert all(size % 4 == 0 for size in sizes)
    assert sum(sizes) == NUM_DIGITS * NUM_FG * NUM_BG * 4
    assert source.counts.sum() == NUM_DIGITS * NUM_FG * NUM_BG * 4
    assert source.counts.min() == 4


def test_each_source_provides_what_it_advertises() -> None:
    loader = one_batch_loader()
    sources = [
        SampleSource(StubModel(), StubAE(), DEVICE, samples_per_combination=2),
        RealSource(StubAE(), loader, DEVICE),
        ReconstructionSource(StubAE(), loader, DEVICE),
        DensitySource(StubModel(), StubAE(), loader, DEVICE),
    ]
    for source in sources:
        batch = next(iter(source.batches()))
        assert source.provides <= batch.provides(), source.name


def test_reconstruction_source_pairs_the_original_with_its_round_trip() -> None:
    source = ReconstructionSource(StubAE(), one_batch_loader(), DEVICE)
    batch = next(iter(source.batches()))

    assert batch.reference is not None and batch.images is not None
    assert batch.reference.shape == batch.images.shape
    # The stub decodes to the mean colour, so a flat input comes back unchanged.
    assert torch.allclose(batch.images, batch.reference, atol=1e-6)


def test_density_source_hands_over_a_usable_score_callable() -> None:
    source = DensitySource(StubModel(), StubAE(), one_batch_loader(), DEVICE)
    batch = next(iter(source.batches()))

    assert batch.log_prob is not None and batch.score is not None
    assert batch.latents is not None
    rescored = batch.score(batch.latents, batch.labels)
    assert torch.allclose(rescored, batch.log_prob)


def test_max_batches_stops_a_data_source_early() -> None:
    loader = one_batch_loader() * 5
    source = RealSource(StubAE(), loader, DEVICE, max_batches=2)
    assert len(list(source.batches())) == 2


def test_a_pass_rejects_a_metric_the_source_cannot_feed() -> None:
    density = DensitySource(StubModel(), StubAE(), one_batch_loader(), DEVICE)
    with pytest.raises(ValueError, match="needs \\['images'\\]"):
        Pass(source=density, metrics=[ColourFidelity()]).validate()

    samples = SampleSource(StubModel(), StubAE(), DEVICE, samples_per_combination=2)
    with pytest.raises(ValueError, match="needs \\['log_prob'\\]"):
        Pass(source=samples, metrics=[NegativeLogLikelihood()]).validate()

    # The pairing the suite is actually built from is accepted.
    Pass(source=samples, metrics=[SampleDiversity(2)]).validate()


def test_a_prefix_keeps_two_passes_of_one_metric_apart() -> None:
    loader = one_batch_loader()
    report = run_suite(
        [
            Pass(
                source=ReconstructionSource(StubAE(), loader, DEVICE),
                metrics=[ColourFidelity()],
                prefix="recon/",
            ),
            Pass(
                source=SampleSource(
                    StubModel(), StubAE(), DEVICE, samples_per_combination=2
                ),
                metrics=[ColourFidelity()],
            ),
        ]
    )
    assert "recon/colour/bg_accuracy" in report.tables
    assert "colour/bg_accuracy" in report.tables


def test_run_suite_needs_at_least_one_metric() -> None:
    source = SampleSource(StubModel(), StubAE(), DEVICE, samples_per_combination=2)
    with pytest.raises(ValueError, match="no metrics given"):
        run_suite([Pass(source=source, metrics=[])])


def test_colour_fidelity_locates_the_foreground_in_the_reference() -> None:
    """A reconstruction that puts the digit somewhere else must not score well.

    Without a reference the metric has to trust the image's own brightest region, so a
    misplaced-but-correctly-coloured patch passes. With the original to locate by, the
    metric looks where the digit actually was and finds background.
    """
    labels = torch.tensor([[0, GREEN, BLACK]])
    original = field(BG_PALETTE[BLACK])
    original[:, :, 9:19, 9:19] = torch.tensor(
        FG_PALETTE[GREEN], dtype=torch.float32
    ).reshape(1, 3, 1, 1)

    misplaced = field(BG_PALETTE[BLACK])
    misplaced[:, :, 22:26, 22:26] = torch.tensor(
        FG_PALETTE[GREEN], dtype=torch.float32
    ).reshape(1, 3, 1, 1)

    unpaired = ColourFidelity()
    unpaired.update(EvalBatch(labels=labels, images=misplaced))
    assert np.nanmax(unpaired.compute().tables["fg_accuracy"]) == 1.0

    paired = ColourFidelity()
    paired.update(
        EvalBatch(labels=labels, images=misplaced, reference=original)
    )
    tables = paired.compute().tables
    assert np.nanmax(tables["fg_accuracy"]) == 0.0
    assert np.nanmin(tables["fg_drift"]) > 0.5


class _NeedsNothing(Metric):
    name = "trivial"
    requires = frozenset()

    def __init__(self) -> None:
        self.seen = 0

    def update(self, batch: EvalBatch) -> None:
        self.seen += batch.size

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, scalars={"seen": float(self.seen)})


def test_a_metric_requiring_nothing_runs_on_any_source() -> None:
    source = RealSource(StubAE(), one_batch_loader(), DEVICE)
    report = run_suite([Pass(source=source, metrics=[_NeedsNothing()])])
    assert report.scalars["trivial/seen"] == 12.0

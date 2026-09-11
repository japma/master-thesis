"""The probes decide whether "this factor lives in these dimensions" is true, so they are
checked against latents where the answer is planted and known: a factor that is only in
its block must read as local and leak-free, and one that is copied outside its block must
be caught leaking.
"""

import numpy as np
import pytest
import torch

from evaluation.latent_probe import blocks_for, encode_dataset, probe_latents
from models.autoencoder import SupervisedVAE, VariationalAutoencoder
from utils.config import AutoencoderConfig, AutoencoderType, SupervisionConfig

CLASSES = 4
DIMS = 6


def planted(count: int, leak: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor 0 lives in dims 0-1, factor 1 in dims 2-3; dims 4-5 are noise.

    With `leak`, factor 1 is also written into dim 5, which is outside its block.
    """
    rng = np.random.default_rng(seed)
    first = rng.integers(0, CLASSES, count)
    second = rng.integers(0, CLASSES, count)

    latents = rng.normal(scale=0.35, size=(count, DIMS))
    latents[:, 0] += 3.0 * first
    latents[:, 1] += -2.0 * first
    latents[:, 2] += 3.0 * second
    latents[:, 3] += 2.0 * second
    if leak:
        latents[:, 5] += 3.0 * second

    return latents, np.stack([first, second], axis=1)


BLOCKS = [slice(0, 2), slice(2, 4)]
NAMES = ["first", "second"]


def report_for(leak: bool):
    train_z, train_y = planted(900, leak, seed=0)
    test_z, test_y = planted(400, leak, seed=1)
    return probe_latents(train_z, train_y, test_z, test_y, BLOCKS, NAMES)


def test_a_factor_in_its_own_block_reads_as_local() -> None:
    first, second = report_for(leak=False).factors

    for probe in (first, second):
        assert probe.sufficiency_linear > 0.95
        assert probe.locality_linear > 0.95
        # Only noise is left outside the block, so nothing beats guessing by much.
        assert probe.exclusivity_linear < probe.majority + 0.15
        assert probe.exclusivity_mlp < probe.majority + 0.15

    assert first.block == (0, 2)
    assert second.block == (2, 4)
    assert first.best_dim in (0, 1)
    assert second.best_dim in (2, 3)


def test_a_factor_copied_outside_its_block_reads_as_leaking() -> None:
    leaking = report_for(leak=True).factors[1]
    clean = report_for(leak=False).factors[1]

    assert leaking.exclusivity_linear > 0.95
    assert leaking.exclusivity_linear > clean.exclusivity_linear + 0.5


def test_mig_is_higher_when_one_dimension_carries_the_factor() -> None:
    rng = np.random.default_rng(0)
    targets = rng.integers(0, CLASSES, 900)[:, None]

    single = rng.normal(scale=0.3, size=(900, DIMS))
    single[:, 0] += 3.0 * targets[:, 0]

    spread = rng.normal(scale=0.3, size=(900, DIMS))
    spread[:, 0] += 3.0 * targets[:, 0]
    spread[:, 1] += 3.0 * targets[:, 0]

    one_dim = probe_latents(single, targets, single, targets, [slice(0, 2)], ["f"])
    two_dims = probe_latents(spread, targets, spread, targets, [slice(0, 2)], ["f"])

    assert one_dim.factors[0].mig > two_dims.factors[0].mig


def test_report_frame_and_active_units() -> None:
    report = report_for(leak=False)
    report.kl_per_dim = np.array([1.0, 0.5, 0.2, 0.001, 0.0, 0.3])

    assert report.active_units == 4
    frame = report.to_frame()
    assert list(frame.index) == NAMES
    assert "leakage (linear, other dims)" in frame.columns


def test_block_count_must_match_the_targets() -> None:
    train_z, train_y = planted(100, leak=False, seed=0)
    with pytest.raises(ValueError, match="blocks named"):
        probe_latents(train_z, train_y, train_z, train_y, BLOCKS[:1], NAMES[:1])


# --- block resolution ---
def ae_config(model_type: AutoencoderType, supervision=None) -> AutoencoderConfig:
    return AutoencoderConfig(
        model_type=model_type,
        latent_dim=10,
        num_blocks=2,
        base_channels=8,
        image_size=8,
        channels=3,
        supervision=supervision,
    )


def test_a_supervised_model_names_its_own_blocks() -> None:
    supervision = SupervisionConfig(
        dims=[4, 3, 2], cardinalities=[10, 6, 3], names=["digit", "fg", "bg"]
    )
    model = SupervisedVAE(config=ae_config(AutoencoderType.SUPERVISED, supervision))

    blocks, names = blocks_for(model, fallback=[slice(0, 1)], fallback_names=["x"])
    assert blocks == [slice(0, 4), slice(4, 7), slice(7, 9)]
    assert names == ["digit", "fg", "bg"]


def test_an_unsupervised_model_falls_back_to_the_given_blocks() -> None:
    model = VariationalAutoencoder(config=ae_config(AutoencoderType.VARIATIONAL))

    blocks, names = blocks_for(model, fallback=BLOCKS, fallback_names=NAMES)
    assert blocks == BLOCKS
    assert names == NAMES


def test_encode_dataset_returns_means_kl_and_targets() -> None:
    torch.manual_seed(0)
    model = VariationalAutoencoder(config=ae_config(AutoencoderType.VARIATIONAL))
    images = torch.rand(6, 3, 8, 8)
    labels = torch.zeros(6, 3, dtype=torch.long)
    loader = [(images, labels)]

    mu, kl, targets = encode_dataset(model, loader, torch.device("cpu"))

    assert mu.shape == (6, 10)
    assert kl.shape == (10,)
    assert targets.shape == (6, 3)

"""The run format, the results table and the sampling loop of the two-stage evaluation."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import Tensor

from dataset_loaders.colour_mnist import all_combinations
from evaluation.generate import sample_latents
from evaluation.models import StdCorrectedSampler
from evaluation.results import result_row, upsert_results
from evaluation.run import (
    RunManifest,
    load_labels,
    load_latents,
    run_dir_name,
    save_run,
)

DEVICE = torch.device("cpu")


def manifest(model_name: str = "cspn_std0.6", seed: int = 0) -> RunManifest:
    return RunManifest(
        model_name=model_name,
        dataset="colour_mnist_uniform_test",
        seed=seed,
        n_samples=180,
        checkpoint_path="psinet_colour_mnist_uniform:v1",
        vae_checkpoint="variational_colour_mnist_uniform:v1",
        git_commit=None,
        std_correction=0.6,
        sampling_seconds=1.25,
        n_parameters=1234,
    )


class LabelEchoModel:
    """Samples a latent that is its label plus a row counter scaled by `std_correction`."""

    def sample(self, labels: Tensor, std_correction: float = 1.0) -> Tensor:
        counter = torch.arange(labels.shape[0], dtype=torch.float32).unsqueeze(1)
        return torch.cat([labels.float(), std_correction * counter], dim=1)


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    original = manifest()
    original.save(tmp_path)
    assert RunManifest.load(tmp_path) == original


def test_manifest_round_trips_a_real_run(tmp_path: Path) -> None:
    real = RunManifest(
        model_name="real",
        dataset="colour_mnist_uniform_test",
        seed=0,
        n_samples=10,
        checkpoint_path=None,
        vae_checkpoint="variational_colour_mnist_uniform:v1",
        git_commit="abc123",
        std_correction=None,
        sampling_seconds=None,
        n_parameters=None,
    )
    real.save(tmp_path)
    assert RunManifest.load(tmp_path) == real


def test_run_keeps_latents_with_their_labels(tmp_path: Path) -> None:
    labels = all_combinations()
    latents = torch.randn(180, 4)
    run_dir = tmp_path / run_dir_name("colour_mnist_uniform_test", "cspn", 3)
    save_run(run_dir, manifest(), latents, labels)

    assert run_dir.name == "colour_mnist_uniform_test__cspn__seed3"
    assert torch.equal(load_latents(run_dir), latents)
    assert torch.equal(load_labels(run_dir), labels)


def test_run_rejects_misaligned_latents_and_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="179 latents but 180 labels"):
        save_run(tmp_path, manifest(), torch.randn(179, 4), all_combinations())


def test_sampler_groups_samples_by_label() -> None:
    sampler = StdCorrectedSampler(LabelEchoModel(), std_correction=0.5)
    y = all_combinations()[:5]
    latents = sampler.sample(y, n_per_label=3)

    assert latents.shape == (5, 3, 4)
    assert torch.equal(latents[:, :, :3], y.float().unsqueeze(1).expand(5, 3, 3))
    assert latents[-1, -1, 3].item() == pytest.approx(0.5 * 14)


def test_sample_latents_labels_every_latent_across_batches() -> None:
    sampler = StdCorrectedSampler(LabelEchoModel(), std_correction=1.0)
    labels = all_combinations()
    latents, sample_labels = sample_latents(
        sampler, labels, n_per_label=2, device=DEVICE, batch_size=64
    )

    assert latents.shape == (360, 4)
    assert torch.equal(sample_labels, labels.repeat_interleave(2, dim=0))
    assert torch.equal(latents[:, :3], sample_labels.float())


def test_upsert_overwrites_the_same_key_and_appends_new_ones(tmp_path: Path) -> None:
    table = tmp_path / "results" / "metrics.csv"

    upsert_results(table, [result_row(manifest(), "fid", 12.0)])
    upsert_results(table, [result_row(manifest(), "fid", 12.0)])
    assert len(pd.read_csv(table)) == 1

    upsert_results(table, [result_row(manifest(), "fid", 10.0)])
    upsert_results(table, [result_row(manifest(seed=1), "fid", 11.0)])
    upsert_results(table, [result_row(manifest("nn_baseline"), "fid", 30.0)])

    frame = pd.read_csv(table)
    assert len(frame) == 3
    assert (
        frame.loc[
            (frame["model"] == "cspn_std0.6") & (frame["seed"] == 0), "value"
        ].item()
        == 10.0
    )
    assert list(frame.columns) == [
        "model",
        "dataset",
        "checkpoint",
        "seed",
        "metric_name",
        "value",
    ]

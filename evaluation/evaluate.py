"""Stage 2: score a run against a reference run, from what stage 1 wrote to disk."""

from pathlib import Path

import torch

from evaluation.fid import decode_batches, frechet_inception_distance
from evaluation.models import load_vae
from evaluation.results import result_row
from evaluation.run import RunManifest, load_latents
from utils.reproducibility import resolve_device

METRICS: tuple[str, ...] = ("fid",)


def evaluate_run(
    run_dir: Path,
    reference_dir: Path,
    metrics: list[str],
    device: torch.device | None = None,
    batch_size: int = 256,
) -> list[dict[str, object]]:
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metrics {unknown}, expected some of {METRICS}")

    run = RunManifest.load(run_dir)
    reference = RunManifest.load(reference_dir)
    if run.dataset != reference.dataset:
        raise ValueError(f"run is on {run.dataset}, reference on {reference.dataset}")
    if run.vae_checkpoint != reference.vae_checkpoint:
        raise ValueError(
            f"run decodes with {run.vae_checkpoint}, "
            f"reference with {reference.vae_checkpoint}"
        )

    device = device if device is not None else resolve_device()
    vae, _ = load_vae(run.vae_checkpoint, device)

    rows: list[dict[str, object]] = []
    if run.sampling_seconds is not None:
        rows.append(result_row(run, "sampling_seconds", run.sampling_seconds))
    if run.n_parameters is not None:
        rows.append(result_row(run, "n_parameters", run.n_parameters))

    if "fid" in metrics:
        fid = frechet_inception_distance(
            decode_batches(
                vae, load_latents(run_dir), device, batch_size, desc="fid: run"
            ),
            decode_batches(
                vae,
                load_latents(reference_dir),
                device,
                batch_size,
                desc="fid: reference",
            ),
            device,
        )
        rows.append(result_row(run, "fid", fid))

    return rows

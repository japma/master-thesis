"""Stage 2: score a cached sample pool. No VAE, no sampling, no gradients.

One CSV per metric under `results/`, accumulating across runs, so plotting a figure is
one `read_csv`. Every file starts with the same columns identifying the run.
"""

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from evaluation.metrics import DIGIT, accuracy, accuracy_by_combination, confusion
from evaluation.samples import (
    SampleManifest,
    load_images,
    load_labels,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
    to_float,
)
from models.classifier import DigitClassifier
from utils.checkpoints import load_classifier_from_path
from utils.config import CheckpointConfig, EvaluationRunConfig
from utils.wandb_utils import download_artifact

DEFAULT_BATCH_SIZE = 512

# Where a set of images came from. Generated is read against the other two.
GENERATED = "generated"
REAL = "real"
RECONSTRUCTION = "reconstruction"

# Identify a run; re-evaluating one replaces its rows rather than duplicating them.
RUN_KEYS = ["checkpoint", "seed", "std_correction"]


def load_judge(
    cfg: CheckpointConfig, device: torch.device
) -> tuple[DigitClassifier, str]:
    """A frozen classifier and the exact `name:vN` it came from."""
    path, checkpoint = download_artifact(cfg.name, cfg.tag)
    model = load_classifier_from_path(path, device=device)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def predict(
    model: DigitClassifier,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    desc: str = "judging",
) -> torch.Tensor:
    """Predicted digit per image, for `(N, C, H, W)` uint8 images."""
    predictions = [
        model(to_float(batch).to(device)).argmax(dim=1).cpu()
        for batch in tqdm(images.split(batch_size), desc=desc)
    ]
    return torch.cat(predictions)


def run_columns(manifest: SampleManifest, classifier: str) -> dict:
    """The identity columns every metric CSV starts with."""
    return {
        "model": manifest.model_type,
        "dataset": manifest.dataset,
        "checkpoint": manifest.model_checkpoint,
        "seed": manifest.seed,
        "std_correction": manifest.std_correction,
        "vae": manifest.vae_checkpoint,
        "classifier": classifier,
    }


def tag(frame: pd.DataFrame, columns: dict, source: str) -> pd.DataFrame:
    """Prefix a metric's own columns with the run identity, so every CSV reads alike."""
    tagged = frame.assign(**columns, source=source)
    identity = [*columns, "source"]
    return tagged[[*identity, *frame.columns]]


def write_metric(path: Path, frame: pd.DataFrame, keys: dict) -> None:
    """Write `frame` to `path`, replacing any rows already there for the same run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        stale = pd.Series(True, index=existing.index)
        for column, value in keys.items():
            stale &= existing[column] == value
        frame = pd.concat([existing[~stale], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


def evaluate_pool(cfg: EvaluationRunConfig, device: torch.device) -> None:
    pool_dir = cfg.pool_dir
    results_root = cfg.evaluation.results_root
    batch_size = cfg.evaluation.batch_size
    if not pool_dir.is_dir():
        raise FileNotFoundError(
            f"No sample pool at {pool_dir}. Run `uv run generate_samples` with this "
            "config first."
        )
    manifest = load_sample_manifest(pool_dir)
    reference_pool = reference_dir(pool_dir)
    reference = load_reference_manifest(reference_pool)
    if reference.vae_checkpoint != manifest.vae_checkpoint:
        raise ValueError(
            f"pool was decoded with {manifest.vae_checkpoint} but its reference pool "
            f"used {reference.vae_checkpoint}; the ceiling would not bound the samples"
        )

    model, classifier = load_judge(cfg.classifier, device)
    num_classes = model.config.num_classes
    columns = run_columns(manifest, classifier)
    keys = {key: columns[key] for key in RUN_KEYS}

    reference_labels = load_labels(reference_pool)
    image_sets = {
        GENERATED: (load_images(pool_dir), load_labels(pool_dir)),
        REAL: (load_originals(reference_pool), reference_labels),
        RECONSTRUCTION: (load_images(reference_pool), reference_labels),
    }

    overall: list[pd.DataFrame] = []
    by_combination: list[pd.DataFrame] = []
    confusions: list[pd.DataFrame] = []
    for source, (images, labels) in image_sets.items():
        predictions = predict(model, images, device, batch_size, desc=source)
        digits = labels[:, DIGIT]
        scores = pd.DataFrame(
            [{"value": accuracy(predictions, digits), "n": int(digits.shape[0])}]
        )
        overall.append(tag(scores, columns, source))
        by_combination.append(
            tag(accuracy_by_combination(predictions, labels), columns, source)
        )
        confusions.append(
            tag(confusion(predictions, digits, num_classes), columns, source)
        )

    write_metric(
        results_root / "digit_accuracy.csv", pd.concat(overall, ignore_index=True), keys
    )
    write_metric(
        results_root / "digit_accuracy_by_combination.csv",
        pd.concat(by_combination, ignore_index=True),
        keys,
    )
    write_metric(
        results_root / "confusion_digit.csv",
        pd.concat(confusions, ignore_index=True),
        keys,
    )
    _print_summary(pd.concat(overall, ignore_index=True))


def _print_summary(scores: pd.DataFrame) -> None:
    print("\ndigit accuracy")
    for row in scores.itertuples(index=False):
        print(f"  {row.source:<16} {row.value:.4f}  (n={row.n})")

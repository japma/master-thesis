"""Stage 2: score a dataset's pool with the judge. No VAE, no sampling, no gradients.

Every model the config lists is scored from its newest generated version (or the one it
pins), against the real images and each model's own VAE round trip. One CSV per metric
under `results/`, accumulating across runs, so plotting a figure is one `read_csv`.
Every file starts with the same columns identifying the run.
"""

from pathlib import Path

import pandas as pd
import torch
from rtpt import RTPT
from tqdm import tqdm

from evaluation.metrics import selected
from evaluation.pools import (
    IMAGES,
    LABELS,
    ModelManifest,
    is_complete,
    load_conditioning,
    load_manifest,
    load_tensor,
    model_root,
    real_dir,
    vae_dir,
    version_number,
)
from evaluation.samples import DIGIT, to_float
from models.classifier import DigitClassifier
from utils.checkpoints import load_classifier_from_path
from utils.config import CheckpointConfig, PoolRunConfig
from utils.progress import batch_count, start_rtpt
from utils.wandb_utils import download_artifact

DEFAULT_BATCH_SIZE = 512

# Where a set of images came from. Generated is read against the other two.
GENERATED = "generated"
REAL = "real"
RECONSTRUCTION = "reconstruction"

# Identify a run; re-evaluating one replaces its rows rather than duplicating them.
RUN_KEYS = ["checkpoint", "seed", "std_correction"]

# A metric is worth printing when it is one number per source; anything with its own
# per-cell columns belongs in the CSV, not the terminal.
SUMMARY_COLUMNS = {"factor", "value", "n"}


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
    rtpt: RTPT | None = None,
) -> torch.Tensor:
    """`(N, num_factors)` predicted classes, laid out like the labels, for `(N, C, H, W)`
    uint8 images."""
    predictions = []
    for batch in tqdm(images.split(batch_size), desc=desc):
        if rtpt is not None:
            rtpt.step(subtitle=desc)
        predictions.append(model.predict(to_float(batch).to(device)).cpu())
    return torch.cat(predictions)


def run_columns(manifest: ModelManifest, classifier: str | None = None) -> dict:
    """The identity columns every metric CSV starts with.

    `classifier` is None where no judge was involved, as in the set metrics.
    """
    return {
        "model": manifest.model_type,
        "dataset": manifest.dataset,
        "checkpoint": manifest.model_checkpoint,
        "seed": manifest.seed,
        "std_correction": manifest.std_correction,
        "vae": manifest.vae_checkpoint,
        "classifier": classifier or "none",
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


def find_models(cfg: PoolRunConfig) -> list[tuple[Path, ModelManifest]]:
    """Every listed model's generated samples, per seed: the pinned version, or the
    newest one generated. Models with nothing generated are reported and left out."""
    found = []
    for seed in cfg.generation.seeds:
        for entry in cfg.models:
            root = model_root(cfg.dataset_dir, seed, entry.type, entry.name)
            std = f"std{entry.std_correction:g}"
            candidates = (
                [root / entry.version]
                if entry.version is not None
                else sorted(root.glob("v*"), key=version_number, reverse=True)
            )
            complete = [v / std for v in candidates if is_complete(v / std)]
            if not complete:
                wanted = f"{entry.name}:{entry.version or 'any version'}"
                print(
                    f"WARNING: nothing generated for {wanted} "
                    f"(std={entry.std_correction:g}, seed={seed}); skipping it"
                )
                continue
            found.append((complete[0], load_manifest(complete[0], ModelManifest)))
    if not found:
        raise FileNotFoundError(
            f"No generated models under {cfg.dataset_dir}. Run `uv run generate_pools` "
            "with this config first."
        )
    return found


def evaluate_pools(cfg: PoolRunConfig, device: torch.device) -> None:
    if cfg.classifier is None:
        raise ValueError(
            f"{cfg.dataset.name} has no `classifier:` in its config, but every metric "
            "here is judged. Score this model with `evaluate_sets` instead."
        )
    dataset_dir = cfg.dataset_dir
    results_root = cfg.evaluation.results_root
    batch_size = cfg.evaluation.batch_size
    models = find_models(cfg)
    real_images = load_tensor(real_dir(dataset_dir), IMAGES)
    real_labels = load_tensor(real_dir(dataset_dir), LABELS)

    model, classifier = load_judge(cfg.classifier, device)
    num_classes = model.config.cardinalities[DIGIT]
    metrics = selected(cfg.evaluation.metrics)
    vae_refs = sorted({manifest.vae_checkpoint for _, manifest in models})
    rtpt = start_rtpt(
        f"evaluate_{cfg.dataset.name}",
        batch_count(real_images.shape[0], batch_size) * (1 + len(vae_refs))
        + sum(batch_count(manifest.n, batch_size) for _, manifest in models),
    )

    def judged(images: torch.Tensor, desc: str) -> tuple[torch.Tensor, torch.Tensor]:
        return to_float(images), predict(model, images, device, batch_size, desc, rtpt)

    real = (*judged(real_images, "real"), real_labels)
    reconstructions = {}
    for vae_ref in vae_refs:
        images = load_tensor(vae_dir(dataset_dir, vae_ref), IMAGES)
        reconstructions[vae_ref] = (*judged(images, vae_ref), real_labels)

    for directory, manifest in models:
        print(
            f"\n=== {manifest.model_checkpoint} (std={manifest.std_correction:g}, "
            f"seed={manifest.seed})"
        )
        columns = run_columns(manifest, classifier)
        keys = {key: columns[key] for key in RUN_KEYS}
        generated = (
            *judged(load_tensor(directory, IMAGES), manifest.model_checkpoint),
            load_conditioning(dataset_dir, manifest.labels),
        )
        sources = {
            GENERATED: generated,
            REAL: real,
            RECONSTRUCTION: reconstructions[manifest.vae_checkpoint],
        }
        frames: dict[str, list[pd.DataFrame]] = {m.FILENAME: [] for m in metrics}
        for source, (pixels, predictions, labels) in sources.items():
            for metric in metrics:
                table = metric.compute(pixels, predictions, labels, num_classes)
                frames[metric.FILENAME].append(tag(table, columns, source))
        tables = {
            filename: pd.concat(parts, ignore_index=True)
            for filename, parts in frames.items()
        }
        for filename, table in tables.items():
            write_metric(results_root / filename, table, keys)
        _print_summary(tables, identity=[*columns, "source"])


def _print_summary(tables: dict[str, pd.DataFrame], identity: list[str]) -> None:
    """Every metric that is one number per source. The per-cell tables are too big to
    read in a terminal and are what the CSVs are for."""
    for filename, table in tables.items():
        own = [column for column in table.columns if column not in identity]
        if not set(own) <= SUMMARY_COLUMNS:
            continue
        print(f"\n{filename.removesuffix('.csv')}")
        for row in table.itertuples(index=False):
            factor = f"{row.factor:<3} " if "factor" in own else ""
            print(f"  {row.source:<16} {factor}{row.value:.4f}  (n={row.n})")

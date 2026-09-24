"""Stage 2: score a dataset's pool with the judge. No VAE, no sampling, no gradients.

Each set of images in the pool is scored once: the real val images, their round trip
through every VAE a listed model uses, and every listed model's newest generated version
(or the one it pins). One CSV per metric under `results/`, accumulating across runs, so
plotting a figure is one `read_csv`. Every file starts with the same columns identifying
the set; a real or reconstruction row leaves the model's columns empty.
"""

from pathlib import Path

import pandas as pd
import torch
from rtpt import RTPT
from tqdm import tqdm

from evaluation.metrics import METRICS
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
from evaluation.samples import to_float
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

# Identify a model's run in the set metrics; re-evaluating one replaces its rows.
RUN_KEYS = ["checkpoint", "seed", "std_correction"]

# With `source`, identify a judged set; re-scoring one replaces its rows.
SET_KEYS = ["dataset", "checkpoint", "seed", "std_correction", "vae"]


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


def reference_columns(dataset: str, classifier: str, vae: str | None = None) -> dict:
    """The identity columns of the real set, or of `vae`'s round trip of it."""
    return {
        "model": None,
        "dataset": dataset,
        "checkpoint": None,
        "seed": None,
        "std_correction": None,
        "vae": vae,
        "classifier": classifier,
    }


def tag(frame: pd.DataFrame, columns: dict, source: str) -> pd.DataFrame:
    """Prefix a metric's own columns with the run identity, so every CSV reads alike."""
    tagged = frame.assign(**columns, source=source)
    identity = [*columns, "source"]
    return tagged[[*identity, *frame.columns]]


def write_metric(path: Path, frame: pd.DataFrame, keys: dict) -> None:
    """Write `frame` to `path`, replacing any rows already there for the same run. An
    empty key matches the rows that leave that column empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        stale = pd.Series(True, index=existing.index)
        for column, value in keys.items():
            stale &= (
                existing[column].isna() if pd.isna(value) else existing[column] == value
            )
        frame = pd.concat([existing[~stale], frame], ignore_index=True)
    frame.astype({"seed": "Int64"}).to_csv(path, index=False)
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
    batch_size = cfg.evaluation.batch_size
    models = find_models(cfg)
    real_images = load_tensor(real_dir(dataset_dir), IMAGES)
    real_labels = load_tensor(real_dir(dataset_dir), LABELS)

    judge, classifier = load_judge(cfg.classifier, device)
    vae_refs = sorted({manifest.vae_checkpoint for _, manifest in models})
    rtpt = start_rtpt(
        f"evaluate_{cfg.dataset.name}",
        batch_count(real_images.shape[0], batch_size) * (1 + len(vae_refs))
        + sum(batch_count(manifest.n, batch_size) for _, manifest in models),
    )

    def score(
        images: torch.Tensor,
        labels: torch.Tensor,
        label_columns: list[int] | None,
        columns: dict,
        source: str,
        desc: str,
    ) -> None:
        """Judge one set, and write every metric's rows for it. `label_columns` are
        the factors the set was conditioned on; None for all of them."""
        print(f"\n=== {source}: {desc}")
        predictions = predict(judge, images, device, batch_size, desc, rtpt)
        label_columns = label_columns or list(range(labels.shape[1]))
        factors = [judge.config.names[c] for c in label_columns]
        cardinalities = [judge.config.cardinalities[c] for c in label_columns]
        keys = {key: columns[key] for key in SET_KEYS} | {"source": source}
        for name in cfg.evaluation.metrics or METRICS:
            table = METRICS[name](
                predictions[:, label_columns],
                labels[:, label_columns],
                factors,
                cardinalities,
            )
            write_metric(
                cfg.evaluation.results_root / f"{name}.csv",
                tag(table, columns, source),
                keys,
            )
            if name == "accuracy":
                for row in table.itertuples(index=False):
                    print(f"  {row.factor:<6} {row.value:.4f}  (n={row.n})")

    score(
        real_images,
        real_labels,
        None,
        reference_columns(cfg.dataset.name, classifier),
        REAL,
        "val split",
    )
    for vae_ref in vae_refs:
        score(
            load_tensor(vae_dir(dataset_dir, vae_ref), IMAGES),
            real_labels,
            None,
            reference_columns(cfg.dataset.name, classifier, vae_ref),
            RECONSTRUCTION,
            vae_ref,
        )
    for directory, manifest in models:
        score(
            load_tensor(directory, IMAGES),
            load_conditioning(dataset_dir, manifest.labels),
            manifest.label_columns,
            run_columns(manifest, classifier),
            GENERATED,
            f"{manifest.model_checkpoint} (std={manifest.std_correction:g}, "
            f"seed={manifest.seed})",
        )

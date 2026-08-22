"""Generate the colour-MNIST dataset from a data-only weight-table config.

The generation rule is a `(digit, foreground, background)` weight table rather than a
hardcoded match statement, so the experimental design is explicit, citable, and can be
swept (a `0.0` cell is a held-out combination; `0.1` is a rare-but-present one).

Three splits are written. Only `train` uses the configured weights; `val` and `test` are
uniform over all 180 combinations, so held-out ones exist as real images to evaluate
against. Seen vs. unseen is then a partition of those sets via the training weight table
(`seen_mask`), not a separate dataset. The training loop validates on `val`, so `test`
stays untouched until final numbers.

    uv run generate_colour_mnist configs/colour_mnist/default.yaml
    uv run generate_colour_mnist configs/colour_mnist/default.yaml --check
    uv run generate_colour_mnist configs/colour_mnist/default.yaml --verify
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps
import tqdm
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torchvision.datasets import MNIST

from dataset_loaders.colour_mnist import (
    BG_COLOURS,
    BG_TO_IDX,
    FG_COLOURS,
    FG_TO_IDX,
    IDX_TO_BG,
    IDX_TO_FG,
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    NUM_BG,
    NUM_DIGITS,
    SPLITS,
    TABLE_SHAPE,
    WEIGHTS_FILENAME,
    build_weight_table,
    load_weight_table,
    save_weight_table,
)

# Independent RNG streams, addressed by index so adding an output later cannot shift the
# draws of an existing one.
_STREAM_TRAIN_VAL_SPLIT = 0
_STREAM_OUTPUT_BASE = 1

# A 4-sigma per-cell threshold over 180 cells expects ~0.01 false alarms, so anything it
# flags is a real deviation rather than sampling noise.
_SIGMA_THRESHOLD = 4.0
# Below this many expected images a cell is too thin for the normal approximation.
_MIN_EXPECTED = 10.0


class Override(BaseModel):
    """One assignment into the weight table. An omitted axis means 'all values'."""

    model_config = ConfigDict(extra="forbid")

    digits: list[int] | None = None
    fg: list[str] | None = None
    bg: list[str] | None = None
    weight: float = Field(ge=0.0)


class GenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    default_weight: float = Field(default=1.0, ge=0.0)
    overrides: list[Override] = Field(default_factory=list)
    data_root: str = "data"
    dataset_dir: str = "colour-mnist"


def _tint_image(
    img: PIL.Image.Image,
    fg_colour: tuple[int, int, int],
    bg_colour: tuple[int, int, int],
) -> PIL.Image.Image:
    img = img.convert("L")
    return PIL.ImageOps.colorize(img, black=bg_colour, white=fg_colour)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratified_split(
    labels: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices per digit so train and val share the same digit distribution."""
    rng = np.random.default_rng([seed, _STREAM_TRAIN_VAL_SPLIT])

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for digit in range(NUM_DIGITS):
        digit_idx = np.flatnonzero(labels == digit)
        shuffled = digit_idx[rng.permutation(digit_idx.size)]
        num_val = round(digit_idx.size * val_fraction)
        val_parts.append(shuffled[:num_val])
        train_parts.append(shuffled[num_val:])

    # Sorted so each split walks the source dataset in its original order.
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def _sample_colours(
    labels: np.ndarray, table: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a (fg, bg) pair per image from its digit's normalized weight row."""
    flat = table.reshape(NUM_DIGITS, -1)
    fg_idx = np.empty(labels.size, dtype=np.int64)
    bg_idx = np.empty(labels.size, dtype=np.int64)

    for digit in range(NUM_DIGITS):
        digit_idx = np.flatnonzero(labels == digit)
        if digit_idx.size == 0:
            continue
        probabilities = flat[digit] / flat[digit].sum()
        draws = rng.choice(flat.shape[1], size=digit_idx.size, p=probabilities)
        fg_idx[digit_idx] = draws // NUM_BG
        bg_idx[digit_idx] = draws % NUM_BG

    return fg_idx, bg_idx


def _write_split(
    name: str,
    dataset: MNIST,
    indices: np.ndarray,
    table: np.ndarray,
    seed: int,
    stream: int,
    root: Path,
) -> dict[str, object]:
    split_dir = root / name
    img_dir = split_dir / "images"
    # Regenerating with fewer images would otherwise leave orphaned PNGs behind.
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    labels = dataset.targets.numpy().astype(np.int64)[indices]
    rng = np.random.default_rng([seed, stream])
    fg_idx, bg_idx = _sample_colours(labels, table, rng)

    csv_path = split_dir / LABELS_FILENAME
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "fg_colour", "bg_colour", "filename"])

        rows = zip(indices, labels, fg_idx, bg_idx, strict=True)
        for out_id, (source_idx, label, fg, bg) in enumerate(
            tqdm.tqdm(rows, total=indices.size, desc=f"{name:9s}")
        ):
            image, _ = dataset[int(source_idx)]
            fg_name = IDX_TO_FG[int(fg)]
            bg_name = IDX_TO_BG[int(bg)]
            tinted = _tint_image(image, FG_COLOURS[fg_name], BG_COLOURS[bg_name])

            filename = f"{out_id:06d}.png"
            tinted.save(img_dir / filename)
            writer.writerow([out_id, int(label), fg_name, bg_name, filename])

    save_weight_table(split_dir / WEIGHTS_FILENAME, table)
    print(f"[{name}] {indices.size} images -> {img_dir}")

    return {"count": int(indices.size), "labels_sha256": _sha256(csv_path)}


def generate(cfg: GenConfig, config_path: Path, root: Path) -> None:
    weighted = build_weight_table(
        cfg.default_weight, [o.model_dump() for o in cfg.overrides]
    )
    uniform = build_weight_table(1.0, [])

    train_source = MNIST(root=cfg.data_root, train=True, download=True)
    test_source = MNIST(root=cfg.data_root, train=False, download=True)

    train_idx, val_idx = _stratified_split(
        train_source.targets.numpy().astype(np.int64), cfg.val_fraction, cfg.seed
    )
    test_idx = np.arange(len(test_source))

    plan: list[tuple[str, MNIST, np.ndarray, np.ndarray]] = [
        ("train", train_source, train_idx, weighted),
        ("val", train_source, val_idx, uniform),
        ("test", test_source, test_idx, uniform),
    ]

    root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, object] = {}
    for offset, (name, source, indices, table) in enumerate(plan):
        splits[name] = _write_split(
            name,
            source,
            indices,
            table,
            cfg.seed,
            _STREAM_OUTPUT_BASE + offset,
            root,
        )

    manifest = {
        "seed": cfg.seed,
        "config_sha256": _sha256(config_path),
        "numpy_version": np.__version__,
        "splits": splits,
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest to {root / MANIFEST_FILENAME}")


def check_distributions(root: Path) -> bool:
    """Compare each split's empirical (digit, fg, bg) frequencies against its weights."""
    ok = True
    for split in SPLITS:
        csv_path = root / split / LABELS_FILENAME
        weights_path = root / split / WEIGHTS_FILENAME
        if not csv_path.exists() or not weights_path.exists():
            print(f"[{split:9s}] MISSING — {csv_path} or {weights_path} not found")
            ok = False
            continue

        table = load_weight_table(weights_path)
        frame = pd.read_csv(csv_path)
        counts = np.zeros(TABLE_SHAPE, dtype=np.float64)
        np.add.at(
            counts,
            (
                frame["label"].to_numpy(dtype=np.int64),
                frame["fg_colour"].map(FG_TO_IDX).to_numpy(dtype=np.int64),
                frame["bg_colour"].map(BG_TO_IDX).to_numpy(dtype=np.int64),
            ),
            1,
        )

        forbidden = 0
        absent = 0
        outliers = 0
        max_deviation = 0.0
        for digit in range(NUM_DIGITS):
            total = counts[digit].sum()
            if total == 0:
                print(f"[{split:9s}] digit {digit} has no images at all")
                ok = False
                continue
            expected = table[digit] / table[digit].sum()
            empirical = counts[digit] / total

            forbidden += int(((expected == 0.0) & (counts[digit] > 0)).sum())
            absent += int(
                (
                    (expected > 0.0)
                    & (total * expected >= _MIN_EXPECTED)
                    & (counts[digit] == 0)
                ).sum()
            )

            deviation = np.abs(empirical - expected)
            sigma = np.sqrt(expected * (1.0 - expected) / total)
            outliers += int((deviation > _SIGMA_THRESHOLD * sigma).sum())
            max_deviation = max(max_deviation, float(deviation.max()))

        seen = int((table > 0.0).sum())
        status = (
            "OK  " if (forbidden == 0 and absent == 0 and outliers == 0) else "FAIL"
        )
        if status == "FAIL":
            ok = False
        print(
            f"[{split:9s}] {status}  {len(frame):>6d} images  "
            f"{seen:>3d}/{table.size} combinations allowed  "
            f"max|emp-exp|={max_deviation:.4f}  "
            f"forbidden-present={forbidden}  expected-absent={absent}  "
            f">{_SIGMA_THRESHOLD:g}sigma={outliers}"
        )

    return ok


def verify_manifest(root: Path, config_path: Path) -> bool:
    """Recompute checksums and compare against the manifest written at generation time."""
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} — regenerate the dataset")
        return False

    manifest = json.loads(manifest_path.read_text())
    ok = True

    config_hash = _sha256(config_path)
    if config_hash != manifest.get("config_sha256"):
        print(
            f"[config   ] MISMATCH  {config_path} does not match the config this "
            "dataset was generated from"
        )
        ok = False
    else:
        print(f"[config   ] OK        {config_path}")

    if manifest.get("numpy_version") != np.__version__:
        print(
            f"[numpy    ] WARN      generated with numpy {manifest.get('numpy_version')}, "
            f"running {np.__version__} — draw streams are only guaranteed within a version"
        )

    for split, info in manifest.get("splits", {}).items():
        csv_path = root / split / LABELS_FILENAME
        if not csv_path.exists():
            print(f"[{split:9s}] MISSING   {csv_path}")
            ok = False
            continue
        actual = _sha256(csv_path)
        if actual == info["labels_sha256"]:
            print(f"[{split:9s}] OK        {info['count']} images")
        else:
            print(
                f"[{split:9s}] MISMATCH  {actual[:16]} != {info['labels_sha256'][:16]}"
            )
            ok = False

    # Image bytes are a deterministic function of (MNIST source digit, colour pair), so a
    # matching labels.csv implies matching pixels for the same verified MNIST download.
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/colour_mnist/default.yaml"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare an existing dataset's empirical distribution to its weight tables",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute checksums of an existing dataset and compare to its manifest",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"No config found at {args.config}")
    cfg = GenConfig.model_validate(yaml.safe_load(args.config.read_text()) or {})
    root = Path(cfg.data_root) / cfg.dataset_dir

    if args.check or args.verify:
        ok = True
        if args.verify:
            ok = verify_manifest(root, args.config) and ok
        if args.check:
            ok = check_distributions(root) and ok
        raise SystemExit(0 if ok else 1)

    generate(cfg, args.config, root)


if __name__ == "__main__":
    main()

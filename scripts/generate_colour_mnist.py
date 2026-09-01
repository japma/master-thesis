"""Generate colour-MNIST: MNIST digits tinted with a (foreground, background) pair."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps
from torchvision.datasets import MNIST
from tqdm import tqdm

from dataset_loaders.colour_mnist import (
    BG_COLOURS,
    DATASET_DIR_NAME,
    FG_COLOURS,
    LABELS_FILENAME,
    NUM_DIGITS,
)

DATA_ROOT: Path = Path("data")
DEFAULT_WEIGHTS: Path = Path("configs/colour_mnist/default.csv")

SEED: int = 42
VAL_FRACTION: float = 0.1


def _load_weights(path: Path) -> pd.DataFrame:
    """Read a `digit,fg,bg,weight` table, requiring one row per combination."""
    frame = pd.read_csv(path)

    missing = {"digit", "fg", "bg", "weight"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    unknown = (set(frame["fg"]) - set(FG_COLOURS)) | (
        set(frame["bg"]) - set(BG_COLOURS)
    )
    if unknown:
        raise ValueError(f"{path} names unknown colours {sorted(unknown)}")

    expected = NUM_DIGITS * len(FG_COLOURS) * len(BG_COLOURS)
    if frame[["digit", "fg", "bg"]].duplicated().any():
        raise ValueError(f"{path} has duplicate (digit, fg, bg) rows")
    if len(frame) != expected:
        raise ValueError(f"{path} has {len(frame)} rows, expected {expected}")
    if (frame["weight"] < 0).any():
        raise ValueError(f"{path} has negative weights")

    empty = frame.groupby("digit")["weight"].sum().loc[lambda s: s <= 0]
    if not empty.empty:
        raise ValueError(f"{path} leaves digits {empty.index.tolist()} with no colours")

    return frame


def _tint_image(
    img: PIL.Image.Image,
    fg_colour: tuple[int, int, int],
    bg_colour: tuple[int, int, int],
) -> PIL.Image.Image:
    return PIL.ImageOps.colorize(img.convert("L"), black=bg_colour, white=fg_colour)


def _stratified_split(
    labels: pd.Series, rng: np.random.Generator
) -> tuple[pd.Index, pd.Index]:
    """Split indices per digit so train and val share the same digit distribution."""
    val = labels.groupby(labels).sample(frac=VAL_FRACTION, random_state=rng).index
    return labels.index.difference(val), val.sort_values()


def _sample_colours(
    labels: pd.Series, weights: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw a (fg, bg) pair per image from its digit's weight row."""
    picks = []
    for digit, group in weights.groupby("digit"):
        index = labels.index[labels == digit]
        drawn = group.sample(
            n=len(index), replace=True, weights=group["weight"], random_state=rng
        )
        picks.append(
            pd.DataFrame(
                {
                    "fg_colour": drawn["fg"].to_numpy(),
                    "bg_colour": drawn["bg"].to_numpy(),
                },
                index=index,
            )
        )
    return pd.concat(picks).sort_index()


def _write_split(
    name: str,
    source: MNIST,
    positions: pd.Index,
    weights: pd.DataFrame,
    rng: np.random.Generator,
    root: Path,
) -> None:
    img_dir = root / name / "images"
    # Regenerating with fewer images would otherwise leave orphaned PNGs behind.
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.Series(source.targets.numpy(), name="label").loc[positions]
    frame = labels.to_frame().join(_sample_colours(labels, weights, rng))
    frame["filename"] = [f"{i:06d}.png" for i in range(len(frame))]

    for row in tqdm(frame.itertuples(), total=len(frame), desc=f"{name:5s}"):
        image, _ = source[int(row.Index)]
        tinted = _tint_image(
            image, FG_COLOURS[row.fg_colour], BG_COLOURS[row.bg_colour]
        )
        tinted.save(img_dir / row.filename)

    frame.to_csv(root / name / LABELS_FILENAME, index=False)
    print(f"[{name:5s}] {len(frame)} images -> {img_dir}")


def generate(weights_path: Path, root: Path) -> None:
    weights = _load_weights(weights_path)
    uniform = weights.assign(weight=1.0)

    train_source = MNIST(root=str(DATA_ROOT), train=True, download=True)
    test_source = MNIST(root=str(DATA_ROOT), train=False, download=True)

    train_labels = pd.Series(train_source.targets.numpy())
    train_idx, val_idx = _stratified_split(
        train_labels, np.random.default_rng([SEED, 0])
    )

    plan = [
        ("train", train_source, train_idx, weights),
        ("val", train_source, val_idx, uniform),
        ("test", test_source, pd.RangeIndex(len(test_source)), uniform),
    ]

    for stream, (name, source, positions, table) in enumerate(plan, start=1):
        rng = np.random.default_rng([SEED, stream])
        _write_split(name, source, positions, table, rng, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        nargs="?",
        default=DEFAULT_WEIGHTS,
        help="weight-table CSV to draw the train split from",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"No weight table found at {args.weights}")
    generate(args.weights, DATA_ROOT / DATASET_DIR_NAME)


if __name__ == "__main__":
    main()

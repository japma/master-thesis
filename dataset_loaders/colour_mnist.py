"""Colour-MNIST dataset and the canonical definition of its label space.

This module is the single source of truth for the palette. The generation script needs
the RGB values, the loader and eval code need the label indices, and both are derived
here from one ordered mapping so they cannot drift apart.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Index order is dict insertion order. Append new colours at the end only — inserting
# one in the middle silently relabels every labels.csv ever generated.
FG_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
    "pink": (255, 0, 255),
}

BG_COLOURS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
}

FG_NAMES: tuple[str, ...] = tuple(FG_COLOURS)
BG_NAMES: tuple[str, ...] = tuple(BG_COLOURS)

FG_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(FG_NAMES)}
BG_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(BG_NAMES)}
IDX_TO_FG: dict[int, str] = dict(enumerate(FG_NAMES))
IDX_TO_BG: dict[int, str] = dict(enumerate(BG_NAMES))

NUM_DIGITS: int = 10
NUM_FG: int = len(FG_NAMES)
NUM_BG: int = len(BG_NAMES)

# Shape of the (digit, foreground, background) generation weight table.
TABLE_SHAPE: tuple[int, int, int] = (NUM_DIGITS, NUM_FG, NUM_BG)
NUM_COMBINATIONS: int = NUM_DIGITS * NUM_FG * NUM_BG

DATASET_DIR_NAME: str = "colour-mnist"
WEIGHTS_FILENAME: str = "weights.json"
LABELS_FILENAME: str = "labels.csv"
MANIFEST_FILENAME: str = "manifest.json"

# Only `train` is drawn from the configured weights. `val` and `test` are uniform over all
# combinations, so held-out ones exist as real images to evaluate against; seen vs. unseen
# is a partition of those sets (see `seen_mask`), not a separate dataset. `val` is what the
# training loop validates on; `test` is not touched until final numbers.
SPLITS: tuple[str, ...] = ("train", "val", "test")


def _resolve_digits(values: Sequence[int] | None) -> list[int]:
    if values is None:
        return list(range(NUM_DIGITS))
    out: list[int] = []
    for value in values:
        if not 0 <= int(value) < NUM_DIGITS:
            raise ValueError(f"digit {value!r} out of range [0, {NUM_DIGITS})")
        out.append(int(value))
    return out


def _resolve_colours(
    values: Sequence[str] | None, to_idx: Mapping[str, int]
) -> list[int]:
    if values is None:
        return list(to_idx.values())
    out: list[int] = []
    for value in values:
        if value not in to_idx:
            raise ValueError(
                f"unknown colour {value!r}; known colours are {sorted(to_idx)}"
            )
        out.append(to_idx[value])
    return out


def build_weight_table(
    default_weight: float = 1.0,
    overrides: Sequence[Mapping[str, object]] | None = None,
) -> np.ndarray:
    """Build the `(10, 6, 3)` sampling-weight table from a data-only spec.

    Every cell starts at `default_weight`; each override assigns its `weight` to the
    cartesian product of the axes it names, with an omitted axis meaning "all values".
    Weights are relative and normalized per digit at sampling time, so `0.0` means
    "never" and `0.1` means "a tenth as often as a `1.0` cell for the same digit".
    """
    table = np.full(TABLE_SHAPE, float(default_weight), dtype=np.float64)

    for override in overrides or []:
        if "weight" not in override:
            raise ValueError(f"override {override!r} is missing 'weight'")
        weight = float(override["weight"])  # type: ignore[arg-type]
        if weight < 0.0:
            raise ValueError(f"negative weight {weight} in override {override!r}")
        digits = _resolve_digits(override.get("digits"))  # type: ignore[arg-type]
        fgs = _resolve_colours(override.get("fg"), FG_TO_IDX)  # type: ignore[arg-type]
        bgs = _resolve_colours(override.get("bg"), BG_TO_IDX)  # type: ignore[arg-type]
        table[np.ix_(digits, fgs, bgs)] = weight

    starved = [d for d in range(NUM_DIGITS) if table[d].sum() <= 0.0]
    if starved:
        raise ValueError(
            f"digits {starved} have zero total weight — no colour combination left to sample"
        )
    return table


def save_weight_table(path: Path, table: np.ndarray) -> None:
    """Write a weight table as self-describing JSON (axis names alongside the values)."""
    payload = {
        "digits": NUM_DIGITS,
        "fg": list(FG_NAMES),
        "bg": list(BG_NAMES),
        "weights": table.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_weight_table(path: Path) -> np.ndarray:
    """Read a weight table written by `save_weight_table`, checking the axes still match."""
    payload = json.loads(path.read_text())
    if list(payload.get("fg", [])) != list(FG_NAMES) or list(
        payload.get("bg", [])
    ) != list(BG_NAMES):
        raise ValueError(
            f"{path} was generated with a different palette "
            f"(fg={payload.get('fg')}, bg={payload.get('bg')}) — regenerate the dataset"
        )
    table = np.asarray(payload["weights"], dtype=np.float64)
    if table.shape != TABLE_SHAPE:
        raise ValueError(f"{path} has shape {table.shape}, expected {TABLE_SHAPE}")
    return table


def load_train_weight_table(root: str | Path) -> np.ndarray:
    """Load the *training* weight table, which is what defines seen vs. unseen.

    `val`/`test` carry uniform tables, so their own `weights.json` says nothing about what
    the model was trained on — always reach for this one when partitioning eval results.
    """
    return load_weight_table(Path(root) / "train" / WEIGHTS_FILENAME)


def seen_mask(table: np.ndarray) -> np.ndarray:
    """Boolean `(10, 6, 3)` mask of combinations the training weights allow to appear.

    Eval code partitions on this to report seen vs. unseen accuracy separately — an
    aggregate metric is dominated by the in-distribution majority and hides the result.
    Under graded (non-0/1) weights the same table also buckets combinations by training
    frequency, and recovers an in-distribution figure from a uniform split by importance
    weighting per-combination losses.
    """
    return table > 0.0


class ColourMNIST(Dataset):
    """MNIST digits tinted with a (foreground, background) colour pair.

    Targets are `[digit, fg_idx, bg_idx]`, indexed by `FG_TO_IDX`/`BG_TO_IDX`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        dataset_dir: str = DATASET_DIR_NAME,
    ):
        self.root = Path(root) / dataset_dir
        self.transform = transform
        self.split = split
        self.split_dir = self.root / split
        self.csv_path = self.split_dir / LABELS_FILENAME

        if not self.csv_path.exists():
            available = (
                sorted(p.name for p in self.root.iterdir() if p.is_dir())
                if self.root.exists()
                else []
            )
            raise FileNotFoundError(
                f"No colour-MNIST split {split!r} at {self.csv_path}. "
                f"Available splits: {available or 'none'}. "
                "Regenerate with `uv run generate_colour_mnist <config>`."
            )

        frame = pd.read_csv(self.csv_path)
        # Materialized up front: indexing a DataFrame per __getitem__ puts pandas in the
        # DataLoader hot path for no benefit.
        self.filenames: list[str] = frame["filename"].astype(str).tolist()
        self.targets: torch.Tensor = torch.tensor(
            np.stack(
                [
                    frame["label"].to_numpy(dtype=np.int64),
                    frame["fg_colour"].map(FG_TO_IDX).to_numpy(dtype=np.int64),
                    frame["bg_colour"].map(BG_TO_IDX).to_numpy(dtype=np.int64),
                ],
                axis=1,
            )
        )

        weights_path = self.split_dir / WEIGHTS_FILENAME
        self.weights: np.ndarray | None = (
            load_weight_table(weights_path) if weights_path.exists() else None
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        img = Image.open(self.split_dir / "images" / self.filenames[index])

        if self.transform:
            img = self.transform(img)

        return img, self.targets[index]

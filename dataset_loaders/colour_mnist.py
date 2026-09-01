"""Colour-MNIST dataset and the canonical definition of its label space."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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

TABLE_SHAPE: tuple[int, int, int] = (NUM_DIGITS, NUM_FG, NUM_BG)
NUM_COMBINATIONS: int = NUM_DIGITS * NUM_FG * NUM_BG

DATASET_DIR_NAME: str = "colour-mnist"
LABELS_FILENAME: str = "labels.csv"

SPLITS: tuple[str, ...] = ("train", "val", "test")

# Each weight table generates its own variant, kept side by side so switching between
# them is a config change rather than a regeneration.
DEFAULT_VARIANT: str = "uniform"


def variant_root(root: str | Path, variant: str = DEFAULT_VARIANT) -> Path:
    """Directory holding one generated colour-MNIST variant."""
    return Path(root) / DATASET_DIR_NAME / variant


def seen_mask(root: str | Path, variant: str = DEFAULT_VARIANT) -> np.ndarray:
    """Boolean `(digit, fg, bg)` mask of the combinations present in the train split.

    Combinations the generation weights zeroed out never appear, so this is what the
    model was actually trained on.
    """
    frame = pd.read_csv(variant_root(root, variant) / "train" / LABELS_FILENAME)
    mask = np.zeros(TABLE_SHAPE, dtype=bool)
    mask[
        frame["label"].to_numpy(dtype=np.int64),
        frame["fg_colour"].map(FG_TO_IDX).to_numpy(dtype=np.int64),
        frame["bg_colour"].map(BG_TO_IDX).to_numpy(dtype=np.int64),
    ] = True
    return mask


class ColourMNIST(Dataset):
    """MNIST digits tinted with a (foreground, background) colour pair.

    Targets are `[digit, fg_idx, bg_idx]`, indexed by `FG_TO_IDX`/`BG_TO_IDX`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        variant: str = DEFAULT_VARIANT,
    ):
        self.root = variant_root(root, variant)
        self.transform = transform
        self.split = split
        self.split_dir = self.root / split
        self.csv_path = self.split_dir / LABELS_FILENAME

        if not self.csv_path.exists():
            if self.root.exists():
                found = sorted(p.name for p in self.root.iterdir() if p.is_dir())
                detail = f"variant {variant!r} has splits {found or 'none'}"
            else:
                base = Path(root) / DATASET_DIR_NAME
                found = (
                    sorted(p.name for p in base.iterdir() if p.is_dir())
                    if base.exists()
                    else []
                )
                detail = f"no variant {variant!r}; generated variants: {found or 'none'}"
            raise FileNotFoundError(
                f"No colour-MNIST split {split!r} at {self.csv_path} — {detail}. "
                "Generate one with "
                "`uv run generate_colour_mnist configs/colour_mnist/<name>.csv`."
            )

        frame = pd.read_csv(self.csv_path)
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

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(
        self, index: int
    ) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        img = Image.open(self.split_dir / "images" / self.filenames[index])

        if self.transform:
            img = self.transform(img)

        return img, self.targets[index]

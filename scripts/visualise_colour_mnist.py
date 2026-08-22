"""Visualize one sample of every (digit, background, foreground) combination.

Produces a 10x18 grid:
    - rows: digit label (0-9)
    - columns: grouped by background (white, black, grey), and within each
      group, ordered by foreground colour according to FG_TO_IDX.
Missing combinations are shown as a black/empty cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dataset_loaders.colour_mnist import (
    BG_TO_IDX,
    DATASET_DIR_NAME,
    FG_TO_IDX,
    IDX_TO_BG,
    IDX_TO_FG,
    LABELS_FILENAME,
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    SPLITS,
)

N_LABELS: int = NUM_DIGITS
N_FG: int = NUM_FG
N_BG: int = NUM_BG
N_COLS: int = N_FG * N_BG  # 18

DATA_ROOT: Path = Path("data") / DATASET_DIR_NAME


def column_index(fg_colour: str, bg_colour: str) -> int:
    """Map (fg, bg) to a column in [0, N_COLS) grouped by background."""
    return BG_TO_IDX[bg_colour] * N_FG + FG_TO_IDX[fg_colour]


def pick_one_sample_per_combination(df: pd.DataFrame) -> dict[tuple[int, int], str]:
    """Return a mapping from (label, column_index) -> filename, one per combo."""
    samples: dict[tuple[int, int], str] = {}
    for row in df.itertuples(index=False):
        label: int = int(row.label)  # type: ignore[attr-defined]
        fg: str = str(row.fg_colour)  # type: ignore[attr-defined]
        bg: str = str(row.bg_colour)  # type: ignore[attr-defined]
        filename: str = str(row.filename)  # type: ignore[attr-defined]
        key: tuple[int, int] = (label, column_index(fg, bg))
        if key not in samples:
            samples[key] = filename
    return samples


def load_image(images_dir: Path, filename: str) -> np.ndarray:
    """Load an image file as a numpy array."""
    return mpimg.imread(images_dir / filename)


def build_grid_figure(
    df: pd.DataFrame,
    images_dir: Path,
    image_shape: tuple[int, int, int] = (28, 28, 3),
) -> Figure:
    """Build the 10x18 overview figure, one axis per (label, combo) cell."""
    samples: dict[tuple[int, int], str] = pick_one_sample_per_combination(df)

    fig: Figure
    axes: np.ndarray
    fig, axes = plt.subplots(
        N_LABELS,
        N_COLS,
        figsize=(N_COLS * 0.9, N_LABELS * 0.9),
        squeeze=False,
    )

    empty_cell: np.ndarray = np.zeros(image_shape, dtype=np.uint8)

    for label in range(N_LABELS):
        for col in range(N_COLS):
            ax: Axes = axes[label][col]
            key: tuple[int, int] = (label, col)
            if key in samples:
                img: np.ndarray = load_image(images_dir, samples[key])
            else:
                img = empty_cell
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if label == 0:
                bg_idx: int = col // N_FG
                fg_idx: int = col % N_FG
                if fg_idx == N_FG // 2:
                    ax.set_title(IDX_TO_BG[bg_idx], fontsize=10, pad=14)
                ax.text(
                    0.5,
                    1.02,
                    IDX_TO_FG[fg_idx][:2],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="grey",
                )
            if col == 0:
                ax.set_ylabel(
                    str(label), fontsize=10, rotation=0, labelpad=15, va="center"
                )

    fig.suptitle("Colour MNIST: sample per (digit, background, foreground)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=SPLITS)
    args = parser.parse_args()

    split_dir: Path = DATA_ROOT / args.split
    df: pd.DataFrame = pd.read_csv(split_dir / LABELS_FILENAME)
    build_grid_figure(df, split_dir / "images")
    plt.show()


if __name__ == "__main__":
    main()

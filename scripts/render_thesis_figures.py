"""Render the sample figures of the thesis as PDFs from the cached pools.

    uv run python scripts/render_thesis_figures.py [OUT_DIR] [--only NAME ...]

OUT_DIR defaults to ../latex/figures/results. Everything is read from what
`generate_pools` already wrote under results/pools/; nothing is sampled here. Only the
latent swaps load a model (the autoencoder, from wandb). The first samples of a label
in pool order are always the ones shown, so every figure is reproducible. A figure
whose pool or checkpoint is missing is skipped and listed in the closing summary.

The label schedules (`*_labels`) and renderers (`render_*`) are shared with
notebooks/thesis_figures.ipynb, which samples the models freshly instead.
"""

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from dataset_loaders.colour_mnist import (
    BG_COLOURS,
    BG_NAMES,
    FG_COLOURS,
    FG_NAMES,
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    all_combinations,
)
from evaluation.generate import load_vae, round_trip
from evaluation.pools import (
    IMAGES,
    LABELS,
    MANIFEST_FILENAME,
    is_complete,
    latest_version,
    load_conditioning,
    load_tensor,
    real_dir,
)
from evaluation.samples import to_float
from models.autoencoder import AbstractAutoencoder
from utils.config import load_dataset_config
from utils.reproducibility import resolve_device

ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "results" / "pools"
WEIGHT_TABLES = ROOT / "configs" / "colour_mnist"
DEFAULT_OUT = ROOT.parent / "latex" / "figures" / "results"

FULL_WIDTH_CM = 16.0
HALF_WIDTH_CM = 7.5
FONT_SIZE = 8
PADDING = 2
CM = 1 / 2.54

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": FONT_SIZE, "pdf.fonttype": 42}
)

MNIST_SAMPLES = 10

# (digit, fg, bg); fg: red green blue cyan yellow pink, bg: white black grey
FEW_LABELS = [(0, 1, 1), (3, 0, 0), (7, 2, 2), (8, 5, 1)]
FEW_SAMPLES = 8

RARE_DIGIT = 0
RARE_COLOURS = [(1, 1), (0, 1), (3, 0), (4, 2)]

SWAP_INPUTS = 6
SWAP_VALUES = {
    "digit": [str(digit) for digit in range(NUM_DIGITS)],
    "fg": list(FG_NAMES),
    "bg": list(BG_NAMES),
}

CELEBA_GRID = 4
# `variational_celeba` version -> the name in the figure file
CELEBA_VAES = {"v4": "beta0p5", "v1": "beta1", "v2": "beta4", "v3": "tcvae"}


class MissingInputError(Exception):
    """An input the figure needs is not on this machine."""


@dataclass
class Pool:
    images: torch.Tensor
    labels: torch.Tensor
    manifest: dict


# --- label schedules: one row per image, in figure order ---
def digit_labels(n: int = MNIST_SAMPLES) -> torch.Tensor:
    """Digits 0-9, `n` times each, as `(10n, 1)`."""
    return torch.arange(NUM_DIGITS).repeat_interleave(n).unsqueeze(1)


def combination_labels() -> torch.Tensor:
    """Every (digit, fg, bg) once: digits as rows, (fg, bg) fg-major as columns."""
    return all_combinations()


def few_labels(n: int = FEW_SAMPLES) -> torch.Tensor:
    """FEW_LABELS, `n` times each."""
    return torch.tensor(FEW_LABELS).repeat_interleave(n, dim=0)


def rare_labels(n: int = FEW_SAMPLES) -> torch.Tensor:
    """RARE_COLOURS of RARE_DIGIT, `n` times each."""
    labels = [(RARE_DIGIT, fg, bg) for fg, bg in RARE_COLOURS]
    return torch.tensor(labels).repeat_interleave(n, dim=0)


# --- reading ---
def read_manifest(directory: Path) -> dict:
    return json.loads((directory / MANIFEST_FILENAME).read_text())


def load_pool(
    dataset: str, model_type: str, name: str, std: float = 1.0, seed: int = 0
) -> Pool:
    """The newest version of a model's samples, with the labels each row was conditioned on."""
    root = POOLS / dataset / f"seed{seed}" / model_type / name
    version = latest_version(root)
    if version is None:
        raise MissingInputError(f"no pool at {root}")
    return load_pool_dir(POOLS / dataset, version / f"std{std:g}")


def load_pool_dir(dataset_dir: Path, directory: Path) -> Pool:
    if not is_complete(directory):
        raise MissingInputError(f"no complete pool at {directory}")
    manifest = read_manifest(directory)
    images = to_float(load_tensor(directory, IMAGES))
    labels = load_conditioning(dataset_dir, manifest["labels"]).long()
    labels = labels.reshape(len(labels), -1)
    if len(images) != len(labels):
        raise ValueError(f"{directory}: {len(images)} images but {len(labels)} labels")
    return Pool(images, labels, manifest)


def load_real(dataset: str) -> Pool:
    directory = real_dir(POOLS / dataset)
    if not is_complete(directory):
        raise MissingInputError(f"no real pool at {directory}")
    images = to_float(load_tensor(directory, IMAGES))
    labels = load_tensor(directory, LABELS).long()
    return Pool(images, labels.reshape(len(labels), -1), read_manifest(directory))


def select(pool: Pool, schedule: torch.Tensor) -> torch.Tensor:
    """The pool's images for a label schedule: for each label, as many of its first
    samples in pool order as the schedule repeats it (labels compared as a prefix)."""
    labels, counts = torch.unique_consecutive(schedule, dim=0, return_counts=True)
    images = []
    for label, count in zip(labels, counts.tolist(), strict=True):
        rows = (pool.labels[:, : len(label)] == label).all(dim=1).nonzero().flatten()
        if len(rows) < count:
            raise MissingInputError(
                f"only {len(rows)} samples of label {tuple(label.tolist())}, need {count}"
            )
        images.append(pool.images[rows[:count]])
    return torch.cat(images)


def combination_frequency(variant: str, digit: int, fg: int, bg: int) -> float:
    """Training frequency of (fg, bg) within `digit`, from the variant's weight table."""
    with open(WEIGHT_TABLES / f"{variant}.csv") as file:
        rows = [row for row in csv.DictReader(file) if int(row["digit"]) == digit]
    total = sum(float(row["weight"]) for row in rows)
    for row in rows:
        if row["fg"] == FG_NAMES[fg] and row["bg"] == BG_NAMES[bg]:
            return float(row["weight"]) / total
    raise ValueError(f"({digit}, {FG_NAMES[fg]}, {BG_NAMES[bg]}) not in {variant}.csv")


@torch.no_grad()
def block_swaps(
    vae: AbstractAutoencoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_inputs: int = SWAP_INPUTS,
) -> dict[str, torch.Tensor]:
    """Per attribute, the first `n_inputs` of `images` (uint8, with `(N, 3)` labels)
    with that attribute's latent block replaced by each value's mean block.

    Blocks are the VAE's `label_slices` (digit, fg, bg); means are over the posterior
    means of all `images`. Each result is row-major: per input the original, its
    reconstruction, then one decode per value.
    """
    latents, reconstructions = round_trip(vae, images, device)
    inputs = latents[:n_inputs]
    swaps = {}
    for k, attribute in enumerate(SWAP_VALUES):
        block = vae.label_slices[k]
        columns = [to_float(images[:n_inputs]), to_float(reconstructions[:n_inputs])]
        for value in range(len(SWAP_VALUES[attribute])):
            swapped = inputs.clone()
            swapped[:, block] = latents[labels[:, k] == value][:, block].mean(dim=0)
            columns.append(vae.decode(swapped.to(device)).float().clamp(0, 1).cpu())
        swaps[attribute] = torch.stack(columns, dim=1).flatten(0, 1)
    return swaps


# --- drawing ---
def grid_of(images: torch.Tensor, columns: int) -> torch.Tensor:
    """`(N, C, H, W)` images, row-major, as one `(3, H', W')` grid."""
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    return make_grid(images.cpu(), nrow=columns, padding=PADDING, pad_value=1.0)


def label_width_cm(labels: Sequence[str]) -> float:
    """Rough width of the widest row label, so the grid plus labels fit the target width."""
    if not labels:
        return 0.0
    return max(len(label) for label in labels) * 0.55 * FONT_SIZE / 72 * 2.54 + 0.15


def save_grid(
    grid: torch.Tensor,
    path: Path,
    width_cm: float,
    cell: tuple[int, int],
    row_labels: Sequence[str] = (),
    column_labels: Sequence[str] = (),
    top: int = 0,
) -> None:
    """Save a `(3, H, W)` grid as a PDF about `width_cm` wide, labels outside the image.

    `cell` is one image's `(height, width)` in pixels; `top` is how many pixel rows of
    the grid sit above the first labelled row (a header).
    """
    height, width = grid.shape[1:]
    image_cm = width_cm - label_width_cm(row_labels)
    figure = plt.figure(figsize=(image_cm * CM, image_cm * height / width * CM))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
    axes.set_axis_off()

    cell_height, cell_width = cell
    for i, label in enumerate(row_labels):
        y = top + PADDING + i * (cell_height + PADDING) + cell_height / 2
        axes.annotate(
            label,
            (0, y),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
        )
    for j, label in enumerate(column_labels):
        x = PADDING + j * (cell_width + PADDING) + cell_width / 2
        axes.annotate(
            label,
            (x, 0),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def swatch(fg: int, bg: int, height: int, width: int) -> torch.Tensor:
    """A `(3, height, width)` patch: the fg colour as a bar on the bg colour, grey rim."""
    patch = torch.tensor(BG_COLOURS[BG_NAMES[bg]], dtype=torch.float32) / 255
    patch = patch.view(3, 1, 1).repeat(1, height, width)
    fg_colour = torch.tensor(FG_COLOURS[FG_NAMES[fg]], dtype=torch.float32) / 255
    patch[:, height // 3 : height - height // 3, width // 4 : width - width // 4] = (
        fg_colour.view(3, 1, 1)
    )
    rim = 0.6
    patch[:, 0, :] = patch[:, -1, :] = patch[:, :, 0] = patch[:, :, -1] = rim
    return patch


def cell_of(images: torch.Tensor) -> tuple[int, int]:
    return int(images.shape[2]), int(images.shape[3])


# --- renderers: images in schedule order -> PDF ---
def render_digit_rows(
    images: torch.Tensor, path: Path, columns: int = MNIST_SAMPLES
) -> None:
    """`digit_labels(columns)` images: one row per digit, labelled."""
    save_grid(
        grid_of(images, columns),
        path,
        FULL_WIDTH_CM,
        cell_of(images),
        row_labels=[str(digit) for digit in range(NUM_DIGITS)],
    )


def render_swaps(images: torch.Tensor, attribute: str, path: Path) -> None:
    """One attribute of `block_swaps`, columns labelled with the values."""
    columns = ["original", "recon."] + SWAP_VALUES[attribute]
    save_grid(
        grid_of(images, len(columns)),
        path,
        FULL_WIDTH_CM,
        cell_of(images),
        column_labels=columns,
    )


def render_combination_grid(images: torch.Tensor, path: Path) -> None:
    """`combination_labels()` images: digits as rows, (fg, bg) as columns under a row
    of colour swatches."""
    pairs = [(fg, bg) for fg in range(NUM_FG) for bg in range(NUM_BG)]
    cell = cell_of(images)
    header = grid_of(
        torch.stack([swatch(fg, bg, cell[0] // 2, cell[1]) for fg, bg in pairs]),
        len(pairs),
    )
    save_grid(
        torch.cat([header, grid_of(images, len(pairs))], dim=1),
        path,
        FULL_WIDTH_CM,
        cell,
        row_labels=[str(digit) for digit in range(NUM_DIGITS)],
        top=header.shape[1],
    )


def render_few_labels(
    images: torch.Tensor, path: Path, columns: int = FEW_SAMPLES
) -> None:
    """`few_labels(columns)` images, one label per row, half width, no text."""
    save_grid(grid_of(images, columns), path, HALF_WIDTH_CM, cell_of(images))


def render_rare(images: torch.Tensor, path: Path, columns: int = FEW_SAMPLES) -> None:
    """`rare_labels(columns)` images, rows labelled with the colouring and its training
    frequency within RARE_DIGIT on the correlated split."""
    row_labels = [
        f"{FG_NAMES[fg]}/{BG_NAMES[bg]} "
        f"({format(100 * combination_frequency('correlated', RARE_DIGIT, fg, bg), '.2g')} %)"
        for fg, bg in RARE_COLOURS
    ]
    save_grid(
        grid_of(images, columns),
        path,
        FULL_WIDTH_CM,
        cell_of(images),
        row_labels=row_labels,
    )


def render_square(images: torch.Tensor, path: Path, n: int = CELEBA_GRID) -> None:
    """The first n x n images, half width, no text."""
    images = images[: n * n]
    save_grid(grid_of(images, n), path, HALF_WIDTH_CM, cell_of(images))


# --- figures from the pools ---
def mnist_samples(out: Path) -> list[Path]:
    """Fig. 1: CSPN on plain MNIST (pool `mnist`, `psinet_mnist`), conditioned on the
    val labels; digits 0-9 (rows) x the first 10 samples of each."""
    pool = load_pool("mnist", "cspn", "psinet_mnist")
    path = out / "mnist_cspn_samples.pdf"
    render_digit_rows(select(pool, digit_labels()), path)
    return [path]


def latent_swaps(
    vae_kind: str, out: Path, dataset: str = "colour_mnist_correlated"
) -> list[Path]:
    """Fig. 2: `block_swaps` in `vae_<dataset>_<vae_kind>` (supervised or anchored),
    the version its reconstruction pool records, over the real val pool."""
    name = f"vae_{dataset}_{vae_kind}"
    version = latest_version(POOLS / dataset / "vaes" / name)
    if version is None:
        raise MissingInputError(
            f"no reconstruction pool for {name}, so no recorded version"
        )
    real = load_real(dataset)
    device = resolve_device()
    try:
        vae, _ = load_vae(
            name, version.name, False, load_dataset_config(dataset), device
        )
    except Exception as error:
        raise MissingInputError(
            f"could not load {name}:{version.name} from wandb ({error})"
        ) from error

    images = load_tensor(real_dir(POOLS / dataset), IMAGES)
    written = []
    for attribute, swapped in block_swaps(vae, images, real.labels, device).items():
        path = out / f"latent_swap_{vae_kind}_{attribute}.pdf"
        render_swaps(swapped, attribute, path)
        written.append(path)
    return written


def combination_grid(
    out: Path,
    model_type: str = "cspn",
    name: str = "cspn_colour_mnist_uniform_anchored",
    dataset: str = "colour_mnist_uniform",
    filename: str = "colour_cspn_anchored_grid.pdf",
) -> list[Path]:
    """Fig. 3: the first sample of every combination. Defaults to the CSPN on the
    AnchoredVAE, uniform split; any colour-MNIST pool works."""
    pool = load_pool(dataset, model_type, name)
    path = out / filename
    render_combination_grid(select(pool, combination_labels()), path)
    return [path]


def few_labels_figure(
    out: Path, model_type: str, name: str, filename: str
) -> list[Path]:
    """Figs. 4 and 6: FEW_LABELS x the first FEW_SAMPLES samples, uniform split."""
    pool = load_pool("colour_mnist_uniform", model_type, name)
    path = out / filename
    render_few_labels(select(pool, few_labels()), path)
    return [path]


def rare_combinations(vae_kind: str, out: Path) -> list[Path]:
    """Fig. 5: CSPN on the correlated split (`cspn_colour_mnist_correlated_<vae_kind>`);
    one frequent and three rare colourings of RARE_DIGIT x the first FEW_SAMPLES."""
    pool = load_pool(
        "colour_mnist_correlated", "cspn", f"cspn_colour_mnist_correlated_{vae_kind}"
    )
    path = out / f"rare_combinations_{vae_kind}.pdf"
    render_rare(select(pool, rare_labels()), path)
    return [path]


def celeba_real(out: Path) -> list[Path]:
    """Fig. 7a: the first CELEBA_GRID^2 images of the CelebA real val pool."""
    path = out / "celeba_real.pdf"
    render_square(load_real("celeba").images, path)
    return [path]


def celeba_cspn(out: Path, seed: int = 0) -> list[Path]:
    """Fig. 7b: every CelebA CSPN pool, the samples conditioned on the first
    CELEBA_GRID^2 real val images, one file per autoencoder (named by CELEBA_VAES from
    the pool's recorded `variational_celeba` version)."""
    root = POOLS / "celeba" / f"seed{seed}" / "cspn"
    pools: dict[str, Path] = {}
    for model in sorted(root.glob("*")) if root.is_dir() else []:
        version = latest_version(model)
        if version is None or not is_complete(version / "std1"):
            continue
        vae_ref = read_manifest(version / "std1")["vae_checkpoint"]
        vae_name, _, vae_version = vae_ref.rpartition(":")
        if vae_name != "variational_celeba" or vae_version not in CELEBA_VAES:
            print(f"celeba_cspn: ignoring {model.name}, decoded by {vae_ref}")
            continue
        kind = CELEBA_VAES[vae_version]
        if kind in pools:
            print(
                f"celeba_cspn: {model.name} and {pools[kind].parent.parent.name} "
                f"both use {vae_ref}; keeping the first"
            )
            continue
        pools[kind] = version / "std1"
    if not pools:
        raise MissingInputError(f"no CelebA CSPN pool below {root}")

    written = []
    for kind, directory in pools.items():
        pool = load_pool_dir(POOLS / "celeba", directory)
        if pool.manifest["labels"] != "real":
            print(
                f"celeba_cspn: skipping {directory}, conditioned on "
                f"{pool.manifest['labels']!r}, not the real labels"
            )
            continue
        path = out / f"celeba_cspn_{kind}.pdf"
        render_square(pool.images, path)
        written.append(path)
    missing = sorted(set(CELEBA_VAES.values()) - set(pools))
    if missing:
        print(f"celeba_cspn: no CSPN pool for {', '.join(missing)}")
    return written


FIGURES: dict[str, Callable[[Path], list[Path]]] = {
    "mnist_samples": mnist_samples,
    "latent_swap_supervised": partial(latent_swaps, "supervised"),
    "latent_swap_anchored": partial(latent_swaps, "anchored"),
    "combination_grid": combination_grid,
    "diversity_cspn": partial(
        few_labels_figure,
        model_type="cspn",
        name="cspn_colour_mnist_uniform_anchored",
        filename="colour_cspn_anchored_diversity.pdf",
    ),
    "diversity_deterministic": partial(
        few_labels_figure,
        model_type="nn_baseline",
        name="nn_baseline_colour_mnist_uniform_variational_deterministic",
        filename="colour_deterministic_variational_diversity.pdf",
    ),
    "diversity_mixture": partial(
        few_labels_figure,
        model_type="nn_baseline",
        name="nn_baseline_colour_mnist_uniform_variational_mixture",
        filename="colour_mixture_variational_diversity.pdf",
    ),
    "rare_combinations_variational": partial(rare_combinations, "variational"),
    "rare_combinations_supervised": partial(rare_combinations, "supervised"),
    "rare_combinations_anchored": partial(rare_combinations, "anchored"),
    "joint_pc_samples": partial(
        few_labels_figure,
        model_type="joint_pc",
        name="joint_pc_colour_mnist_uniform_variational",
        filename="joint_pc_uniform_samples.pdf",
    ),
    "cspn_variational_samples": partial(
        few_labels_figure,
        model_type="cspn",
        name="cspn_colour_mnist_uniform_variational",
        filename="cspn_variational_uniform_samples.pdf",
    ),
    "celeba_real": celeba_real,
    "celeba_cspn": celeba_cspn,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURES),
        metavar="NAME",
        help=f"render only these: {', '.join(FIGURES)}",
    )
    args = parser.parse_args()

    written: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for name in args.only or FIGURES:
        print(f"--- {name}")
        try:
            files = FIGURES[name](args.out_dir)
        except MissingInputError as reason:
            skipped.append((name, str(reason)))
            continue
        written += files

    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    print("Skipped:" if skipped else "Skipped: none")
    for name, reason in skipped:
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()

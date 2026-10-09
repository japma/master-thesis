"""Render placeholder sample grids for the thesis from the cached pools.

    uv run python scripts/render_thesis_grids.py [OUT_DIR]

OUT_DIR defaults to ../latex/figures/results. Reads only what generate_pools already
wrote; nothing is sampled here.
"""

import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

POOL = Path("results/pools/colour_mnist_uniform")
CELEBA = Path("results/samples/celeba__psinet_celeba__seed0")

# name in the figure file -> pool directory holding images.pt
COLOUR_MODELS = {
    "cspn_variational": "seed0/cspn/cspn_colour_mnist_uniform_variational",
    "cspn_anchored": "seed0/cspn/cspn_colour_mnist_uniform_anchored",
    "mixture_variational": "seed0/nn_baseline/nn_baseline_colour_mnist_uniform_variational_mixture",
    "deterministic_variational": "seed0/nn_baseline/nn_baseline_colour_mnist_uniform_variational_deterministic",
    "joint_pc_variational": "seed0/joint_pc/joint_pc_colour_mnist_uniform_variational",
}

# (digit, fg, bg) indices; fg: red green blue cyan yellow pink, bg: white black grey
DIVERSITY_LABELS = [(0, 1, 1), (3, 0, 0), (7, 2, 2), (8, 5, 1)]
SAMPLES_PER_LABEL = 8


def load_images(directory: Path) -> torch.Tensor:
    """images.pt of the newest version/std directory below `directory`."""
    candidates = sorted(directory.rglob("images.pt"))
    if not candidates:
        raise FileNotFoundError(f"no images.pt below {directory}")
    return to_unit(torch.load(candidates[-1], map_location="cpu"))


def to_unit(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    return images.float().clamp(0.0, 1.0)


def rows_for(labels: torch.Tensor, label: tuple[int, int, int]) -> torch.Tensor:
    return (labels == torch.tensor(label)).all(dim=1).nonzero().flatten()


def combination_grid(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """One sample per cell: rows are digits, columns the 18 (fg, bg) pairs."""
    cells = [
        images[rows_for(labels, (digit, fg, bg))[0]]
        for digit in range(10)
        for fg in range(6)
        for bg in range(3)
    ]
    return torch.stack(cells)


def diversity_grid(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """SAMPLES_PER_LABEL samples for each of DIVERSITY_LABELS, one label per row."""
    return torch.cat(
        [images[rows_for(labels, label)[:SAMPLES_PER_LABEL]] for label in DIVERSITY_LABELS]
    )


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../latex/figures/results")
    out.mkdir(parents=True, exist_ok=True)

    labels = torch.load(POOL / "stratified_100" / "labels.pt", map_location="cpu").long()
    for name, directory in COLOUR_MODELS.items():
        try:
            images = load_images(POOL / directory)
        except FileNotFoundError as error:
            print(f"skipping {name}: {error}")
            continue
        save_image(combination_grid(images, labels), out / f"colour_{name}_grid.png", nrow=18, padding=1)
        save_image(diversity_grid(images, labels), out / f"colour_{name}_diversity.png", nrow=SAMPLES_PER_LABEL, padding=1)
        print(f"wrote {name}")

    if (CELEBA / "images.pt").exists():
        images = to_unit(torch.load(CELEBA / "images.pt", map_location="cpu"))
        save_image(images[:32], out / "celeba_cspn_samples.png", nrow=8, padding=1)
        originals = CELEBA / "reference" / "originals.pt"
        if originals.exists():
            real = to_unit(torch.load(originals, map_location="cpu"))
            save_image(real[:32], out / "celeba_real.png", nrow=8, padding=1)
        print("wrote celeba")


if __name__ == "__main__":
    main()

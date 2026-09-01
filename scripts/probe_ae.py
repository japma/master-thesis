"""Does the autoencoder represent colour combinations it never saw in training?

This is the diagnostic that decides where the compositional failure lives. Held-out
combinations (digit 1 only ever on black, digit 2 only on white, digit 0 only in
red/green/blue) exist as real images in the uniform `test` split, so the autoencoder can be
asked about them directly.

Three readings, because the first alone cannot distinguish the two hypotheses:

  reconstruction error   per combination, with per-axis marginals as the control
  colour fidelity        does the reconstruction keep the intended fg/bg colour
  latent geometry        do held-out latents sit inside the distribution the CSPN trained on

    AE reconstructs held-out badly              -> entanglement; supervised latents are the fix
    reconstructs fine, latents in-distribution  -> representable and reachable; the failure is
                                                   the hypernetwork's conditioning
    reconstructs fine, latents off on their own -> a region the CSPN never saw; supervised
                                                   latents help by making it label-predictable

    uv run probe_ae                       # variational_colour_mnist:latest on the test split
    uv run probe_ae --name X --tag v3     # a specific artifact
    uv run probe_ae --no-figures          # numbers only
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loaders.colour_mnist import (
    NUM_DIGITS,
    ColourMNIST,
    seen_mask,
)
from utils import resolve_device
from utils.checkpoints import load_ae_from_path
from utils.probes import (
    latent_mahalanobis,
    marginals,
    per_image_seen,
    run_combination_probe,
    stack_images,
    weighted_mean,
)
from utils.visualisation import plot_combination_heatmap, show_comparison
from utils.wandb_utils import load_from_wandb

DATA_ROOT = Path("data")


def _split_line(
    name: str, values: np.ndarray, mask: np.ndarray, fmt: str = "8.5f"
) -> None:
    """One row comparing trained-on combinations against held-out ones."""
    seen_value = float(values[mask].mean())
    unseen_value = float(values[~mask].mean())
    ratio = f"{unseen_value / seen_value:5.2f}x" if seen_value else "    n/a"
    print(
        f"  {name:26s} seen {seen_value:{fmt}}   unseen {unseen_value:{fmt}}   "
        f"ratio {ratio}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="variational_colour_mnist")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    device = resolve_device()
    ae = load_ae_from_path(load_from_wandb(args.name, args.tag), device=device).to(
        device
    )

    dataset = ColourMNIST(
        root=DATA_ROOT,
        split=args.split,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    seen = seen_mask(DATA_ROOT / "colour-mnist")

    print(
        f"\n{args.name}:{args.tag} on '{args.split}' | {len(dataset)} images | "
        f"device={device}\n{int(seen.sum())}/{seen.size} combinations were trained on\n"
    )

    probe = run_combination_probe(ae, loader, device)
    image_seen = per_image_seen(probe.targets, seen)

    print("per-image, split by whether the combination was trained on")
    _split_line("reconstruction MSE", probe.per_image_error, image_seen)
    _split_line("background colour correct", probe.bg_hit, image_seen, fmt="8.3f")
    _split_line("background colour drift", probe.bg_drift, image_seen, fmt="8.4f")
    _split_line("foreground colour correct", probe.fg_hit, image_seen, fmt="8.3f")
    _split_line("foreground colour drift", probe.fg_drift, image_seen, fmt="8.4f")

    distance = latent_mahalanobis(probe.latents, image_seen)
    _split_line("latent Mahalanobis", distance, image_seen, fmt="8.3f")

    print(
        "\nwithin-digit: each digit's held-out cells against its own trained cells\n"
        "  (the aggregate above is confounded — only digits 0, 1, 2 have holdouts, so it\n"
        "   compares those digits against 3-9 rather than held-out against trained)"
    )
    for digit in range(NUM_DIGITS):
        held = ~seen[digit]
        if not held.any():
            continue
        for label, table in (
            ("reconstruction MSE", probe.error[digit]),
            ("background drift", probe.bg_drift_table[digit]),
            ("foreground drift", probe.fg_drift_table[digit]),
        ):
            trained_value = weighted_mean(table, seen[digit], probe.counts[digit])
            held_value = weighted_mean(table, held, probe.counts[digit])
            print(
                f"  digit {digit}  {label:20s} trained {trained_value:8.5f}   "
                f"held-out {held_value:8.5f}   ratio {held_value / trained_value:6.2f}x"
            )

    print(
        "\nreference: the same metrics for digits with no holdout, all cells trained\n"
        "  (a held-out value inside this range is normal, whatever its ratio says)"
    )
    for label, table in (
        ("reconstruction MSE", probe.error),
        ("background drift", probe.bg_drift_table),
    ):
        values = [
            weighted_mean(table[d], seen[d], probe.counts[d])
            for d in range(NUM_DIGITS)
            if seen[d].all()
        ]
        print(f"  {label:22s} {min(values):.5f} - {max(values):.5f}")

    print("\ncontrol: marginals of reconstruction MSE along each axis alone")
    axis_marginals = marginals(probe.error, probe.counts)
    print("  by digit:", np.array2string(axis_marginals["digit"], precision=5))
    print("  by fg   :", np.array2string(axis_marginals["fg"], precision=5))
    print("  by bg   :", np.array2string(axis_marginals["bg"], precision=5))

    print("\nheld-out combinations, worst reconstruction first")
    held = np.argwhere(~seen)
    order = np.argsort([-probe.error[tuple(c)] for c in held])
    for digit, fg, bg in held[order][:8]:
        print(
            f"  digit {digit}  fg {fg}  bg {bg}   mse {probe.error[digit, fg, bg]:.5f}   "
            f"bg-colour correct {probe.bg_accuracy[digit, fg, bg]:.2f}   "
            f"fg-colour correct {probe.fg_accuracy[digit, fg, bg]:.2f}"
        )

    if args.no_figures:
        return

    import matplotlib.pyplot as plt

    plot_combination_heatmap(probe.error, ~seen, "reconstruction MSE")
    plot_combination_heatmap(probe.bg_drift_table, ~seen, "background colour drift")
    plot_combination_heatmap(probe.fg_drift_table, ~seen, "foreground colour drift")

    worst = held[order][0]
    picks = np.flatnonzero(
        (probe.targets[:, 0] == worst[0])
        & (probe.targets[:, 1] == worst[1])
        & (probe.targets[:, 2] == worst[2])
    )[:8]
    originals = stack_images(dataset, picks).to(device)
    with torch.no_grad():
        recons = ae.decode(ae.encode(originals))
    show_comparison(
        originals.cpu(),
        recons.cpu(),
        title=f"worst held-out combination: digit {worst[0]}, fg {worst[1]}, bg {worst[2]}",
    )
    plt.show()


if __name__ == "__main__":
    main()

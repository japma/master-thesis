"""Can the CSPN *generate* the colour combinations it was never trained on?

`probe_ae` establishes that the autoencoder represents held-out combinations fine — real
images of them reconstruct with background drift inside the normal range, and their latents
sit inside the training distribution. This script asks the other half: conditioned on a
held-out (digit, fg, bg) triple, does the circuit actually produce it?

  generated colours right, like reconstruction -> conditioning generalizes; the failure is
                                                  somewhere else entirely
  generated colours wrong, reconstruction fine -> the hypernetwork cannot reach a latent
                                                  region that demonstrably exists
  contrast collapses on held-out combinations  -> not a wrong colour but a flat image; the
                                                  circuit has no mass there at all

Read the within-digit table, not the aggregate: only digits 0, 1 and 2 have holdouts, so an
aggregate comparison is really digits 0-2 against digits 3-9.

    uv run probe_cspn
    uv run probe_cspn --samples 128 --std-correction 0.8
"""

import argparse

import numpy as np
import torch

from dataset_loaders.colour_mnist import (
    DEFAULT_VARIANT,
    NUM_DIGITS,
    seen_mask,
)
from evaluation import run_generation_probe, weighted_mean
from utils import resolve_device
from utils.checkpoints import (
    load_ae_from_path,
    load_cspn_from_path,
    load_joint_pc_from_path,
    load_nn_baseline_from_path,
)
from utils.visualisation import plot_combination_heatmap, show
from utils.wandb_utils import load_from_wandb

DATA_ROOT = "data"

# Every one of these exposes sample(labels, std_correction) over the same latent space,
# so the probe treats them interchangeably.
MODEL_LOADERS = {
    "cspn": load_cspn_from_path,
    "joint_pc": load_joint_pc_from_path,
    "nn_baseline": load_nn_baseline_from_path,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cspn", default="psinet_colour_mnist")
    parser.add_argument("--ae", default="variational_colour_mnist")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--std-correction", type=float, default=1.0)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--model-type",
        default="cspn",
        choices=sorted(MODEL_LOADERS),
        help="which checkpoint format --cspn names; all are probed identically",
    )
    args = parser.parse_args()

    device = resolve_device()
    load_model = MODEL_LOADERS[args.model_type]
    cspn = load_model(load_from_wandb(args.cspn, args.tag), device=device).to(device)
    ae = load_ae_from_path(load_from_wandb(args.ae, args.tag), device=device).to(device)

    seen = seen_mask(DATA_ROOT, args.variant)
    counts = np.full(seen.shape, float(args.samples))

    print(
        f"\n{args.model_type} {args.cspn}:{args.tag} -> {args.ae}:{args.tag} | "
        f"{args.samples} samples per combination | "
        f"std_correction={args.std_correction} | device={device}\n"
    )

    probe = run_generation_probe(
        cspn,
        ae,
        device,
        samples_per_combination=args.samples,
        std_correction=args.std_correction,
    )

    print("within-digit: generated held-out combinations vs that digit's trained ones")
    for digit in range(NUM_DIGITS):
        held = ~seen[digit]
        if not held.any():
            continue
        for label, table in (
            ("background correct", probe.bg_accuracy[digit]),
            ("foreground correct", probe.fg_accuracy[digit]),
            ("fg/bg contrast", probe.contrast_table[digit]),
        ):
            trained_value = weighted_mean(table, seen[digit], counts[digit])
            held_value = weighted_mean(table, held, counts[digit])
            print(
                f"  digit {digit}  {label:20s} trained {trained_value:7.3f}   "
                f"held-out {held_value:7.3f}   ratio {held_value / trained_value:6.2f}x"
            )
        print()

    print("reference: digits with no holdout, every combination trained")
    for label, table in (
        ("background correct", probe.bg_accuracy),
        ("foreground correct", probe.fg_accuracy),
        ("fg/bg contrast", probe.contrast_table),
    ):
        values = [
            weighted_mean(table[d], seen[d], counts[d])
            for d in range(NUM_DIGITS)
            if seen[d].all()
        ]
        print(f"  {label:20s} {min(values):.3f} - {max(values):.3f}")

    print("\nworst generated combinations by background accuracy")
    order = np.argsort(probe.bg_accuracy, axis=None)
    for flat in order[:10]:
        digit, fg, bg = np.unravel_index(flat, probe.bg_accuracy.shape)
        mark = "held-out" if not seen[digit, fg, bg] else "trained "
        print(
            f"  digit {digit}  fg {fg}  bg {bg}  [{mark}]  "
            f"bg {probe.bg_accuracy[digit, fg, bg]:.2f}  "
            f"fg {probe.fg_accuracy[digit, fg, bg]:.2f}  "
            f"contrast {probe.contrast_table[digit, fg, bg]:.3f}"
        )

    if args.no_figures:
        return

    import matplotlib.pyplot as plt

    plot_combination_heatmap(
        probe.bg_accuracy, ~seen, "generated background correct", vmin=0.0, vmax=1.0
    )
    plot_combination_heatmap(
        probe.fg_accuracy, ~seen, "generated foreground correct", vmin=0.0, vmax=1.0
    )
    plot_combination_heatmap(probe.contrast_table, ~seen, "generated fg/bg contrast")

    with torch.no_grad():
        held = np.argwhere(~seen)
        labels = torch.tensor(
            [list(c) for c in held for _ in range(2)], dtype=torch.long, device=device
        )
        images = ae.decode(cspn.sample(labels, std_correction=args.std_correction))
    show(images[:32].cpu(), title="generated held-out combinations")
    plt.show()


if __name__ == "__main__":
    main()

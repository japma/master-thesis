"""Run the evaluation suite on any latent-space model.

Anything exposing `sample(labels, std_correction)` and `forward(z, labels)` is fair
game -- CSPN, JointPC, and both neural baselines -- so the numbers are comparable
across model families by construction.

    uv run eval_model --model cspn --name psinet_colour_mnist
    uv run eval_model --model nn_baseline --name nn_baseline_colour_mnist_uniform_mixture
    uv run eval_model --model joint_pc --name joint_pc_colour_mnist --variant skewed
    uv run eval_model --model cspn --name X --skip digit   # no classifier trained yet

Reads the held-out mask from the variant's train split, so every table is reported
split into the combinations the model saw and the ones it never did.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loaders.colour_mnist import DEFAULT_VARIANT, ColourMNIST, seen_mask
from evaluation import (
    ColourFidelity,
    DigitAccuracy,
    LabelDiscrimination,
    LatentPlausibility,
    Metric,
    NegativeLogLikelihood,
    SampleDiversity,
    load_digit_classifier,
    run_eval_suite,
)
from utils import resolve_device
from utils.checkpoints import (
    load_ae_from_path,
    load_cspn_from_path,
    load_joint_pc_from_path,
    load_nn_baseline_from_path,
    read_source_artifact,
)
from utils.wandb_utils import download_artifact, load_from_wandb, trained_with

DATA_ROOT = "data"

MODEL_LOADERS = {
    "cspn": load_cspn_from_path,
    "joint_pc": load_joint_pc_from_path,
    "nn_baseline": load_nn_baseline_from_path,
}

ALL_METRICS = ("colour", "digit", "diversity", "latent", "nll", "discrimination")


def build_loader(variant: str, split: str, batch_size: int) -> DataLoader:
    dataset = ColourMNIST(
        root=DATA_ROOT,
        split=split,
        variant=variant,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def reference_latents(ae, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Real train-split latents, for the off-manifold check."""
    ae.eval()
    latents = [ae.encode(images.to(device)).cpu().numpy() for images, _ in loader]
    return np.concatenate(latents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LOADERS))
    parser.add_argument("--name", required=True, help="checkpoint artifact name")
    parser.add_argument(
        "--ae",
        default=None,
        help="autoencoder artifact; by default the one this model was trained with, "
        "resolved from wandb lineage and then from the checkpoint itself",
    )
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--std-correction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--density-batches",
        type=int,
        default=8,
        help="batches of real data for nll/discrimination (discrimination scores all "
        "180 combinations per batch, so this is the expensive knob)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=ALL_METRICS, help="metrics to omit"
    )
    args = parser.parse_args()

    device = resolve_device()
    wanted = [m for m in ALL_METRICS if m not in args.skip]

    model_path, _ = download_artifact(args.name, args.tag)
    model = MODEL_LOADERS[args.model](model_path, device=device).to(device)

    # Scoring a model against an autoencoder other than the one that produced its
    # training latents is silent -- the latent dims still line up -- so the pairing is
    # looked up rather than defaulted.
    ae_artifact = (
        args.ae
        or trained_with(args.name, args.tag)
        or read_source_artifact(model_path)
    )
    if ae_artifact is None:
        parser.error(
            f"could not work out which autoencoder {args.name}:{args.tag} was trained "
            "with -- pass --ae explicitly"
        )
    print(f"Autoencoder: {ae_artifact}")
    ae = load_ae_from_path(
        load_from_wandb(ae_artifact, args.tag), device=device
    ).to(device)

    test_loader = build_loader(args.variant, args.split, args.batch_size)

    sample_metrics: list[Metric] = []
    if "colour" in wanted:
        sample_metrics.append(ColourFidelity())
    if "digit" in wanted:
        sample_metrics.append(DigitAccuracy(load_digit_classifier(device=device)))
    if "diversity" in wanted:
        sample_metrics.append(SampleDiversity(args.samples))
    if "latent" in wanted:
        train_loader = build_loader(args.variant, "train", args.batch_size)
        sample_metrics.append(
            LatentPlausibility(reference_latents(ae, train_loader, device))
        )

    density_metrics: list[Metric] = []
    if "nll" in wanted:
        density_metrics.append(NegativeLogLikelihood())
    if "discrimination" in wanted:
        density_metrics.append(LabelDiscrimination())

    print(
        f"\n{args.model} {args.name}:{args.tag} -> {ae_artifact} | "
        f"{args.variant}/{args.split} | {args.samples} samples per combination | "
        f"std_correction={args.std_correction} | device={device}\n"
    )

    report = run_eval_suite(
        model,
        ae,
        device,
        sample_metrics=sample_metrics,
        density_metrics=density_metrics,
        loader=test_loader,
        seen=seen_mask(DATA_ROOT, args.variant),
        samples_per_combination=args.samples,
        std_correction=args.std_correction,
        max_density_batches=args.density_batches,
    )

    print(f"{'metric':<28} {'overall':>10} {'trained':>10} {'held-out':>10}")
    print("-" * 62)
    for name in sorted(report.tables):
        trained, held_out = report.split(name)
        print(
            f"{name:<28} {report.overall(name):>10.4f} {trained:>10.4f} "
            f"{held_out:>10.4f}"
        )

    if report.scalars:
        print()
        for name, value in sorted(report.scalars.items()):
            print(f"{name:<28} {value:>10.4f}")

    print("\nper-axis marginals (the control: is a gap the digit, the fg, or the bg?)")
    for name in sorted(report.tables):
        axes = report.marginals(name)
        parts = " ".join(
            f"{axis}[{np.nanmin(values):.3f}-{np.nanmax(values):.3f}]"
            for axis, values in axes.items()
        )
        print(f"  {name:<26} {parts}")


if __name__ == "__main__":
    main()

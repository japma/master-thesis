"""Run the evaluation on any latent-space model.

Anything exposing `sample(labels, std_correction)` and `forward(z, labels)` is fair
game -- CSPN, JointPC, and both neural baselines -- so the numbers are comparable
across model families by construction.

    uv run eval_model --model cspn --name psinet_colour_mnist
    uv run eval_model --model nn_baseline --name nn_baseline_colour_mnist_uniform_mixture
    uv run eval_model --model joint_pc --name joint_pc_colour_mnist --variant skewed
    uv run eval_model --model cspn --name X --output results/X   # also write the frames

Samples are printed next to real test images (the reference) and their reconstructions
(the ceiling the autoencoder allows).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loaders.colour_mnist import DEFAULT_VARIANT, ColourMNIST, seen_mask
from evaluation import evaluate_model, load_digit_classifier, predicted_digit_entropy
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

IMAGE_COLUMNS = [
    "bg_accuracy",
    "fg_accuracy",
    "bg_drift",
    "fg_drift",
    "contrast",
    "digit_accuracy",
    "digit_confidence",
    "digit_entropy",
    "mahalanobis",
]
SPREAD_COLUMNS = ["pixel_std", "latent_std"]
DENSITY_COLUMNS = [
    "nll",
    "joint_label_accuracy",
    "digit_label_accuracy",
    "fg_label_accuracy",
    "bg_label_accuracy",
]
SOURCES = ["sample", "reconstruction", "real"]


def build_loader(variant: str, split: str, batch_size: int) -> DataLoader:
    dataset = ColourMNIST(
        root=DATA_ROOT,
        split=split,
        variant=variant,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


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
        "--density-images",
        type=int,
        default=2048,
        help="real images scored for nll and label accuracy; each costs 180 model "
        "evaluations, so this is the expensive knob",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=None, help="directory to write the frames to"
    )
    args = parser.parse_args()

    device = resolve_device()

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
    ae = load_ae_from_path(
        load_from_wandb(ae_artifact, args.tag), device=device
    ).to(device)
    judge = load_digit_classifier(device=device)
    seen = seen_mask(DATA_ROOT, args.variant)

    print(
        f"\n{args.model} {args.name}:{args.tag} -> {ae_artifact} | "
        f"{args.variant}/{args.split} | {args.samples} samples per combination | "
        f"std_correction={args.std_correction} | seed={args.seed} | device={device}\n"
    )

    torch.manual_seed(args.seed)
    evaluation = evaluate_model(
        model,
        ae,
        judge,
        train_loader=build_loader(args.variant, "train", args.batch_size),
        test_loader=build_loader(args.variant, args.split, args.batch_size),
        device=device,
        seen=seen,
        samples_per_combination=args.samples,
        std_correction=args.std_correction,
        density_images=args.density_images,
    )
    images = evaluation.images
    samples = images[images["source"] == "sample"]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 4)

    print("images (digit judge ceiling: 0.971 on real images, floor 0.0996 on noise)")
    table = images.groupby("source")[IMAGE_COLUMNS].mean().T[SOURCES]
    table.loc["predicted_digit_entropy"] = [
        predicted_digit_entropy(images[images["source"] == source])
        for source in SOURCES
    ]
    print(table, "\n")

    print(
        "spread within a combination (read next to colour fidelity, never without it)"
    )
    spread = evaluation.spread.groupby("source")[SPREAD_COLUMNS].mean().T
    print(spread[["sample", "real"]], "\n")

    print(f"density on {len(evaluation.density)} real {args.split} images")
    print(evaluation.density[DENSITY_COLUMNS].mean().to_string(), "\n")

    # Only digits with a held-out combination can be compared fairly: an aggregate
    # trained-vs-held-out split compares those digits against all the others.
    held_out_digits = sorted(samples.loc[~samples["seen"], "digit"].unique())
    if held_out_digits:
        print("samples, trained vs held-out within each digit that has holdouts")
        within = samples[samples["digit"].isin(held_out_digits)]
        print(within.groupby(["digit", "seen"])[IMAGE_COLUMNS].mean().T, "\n")
    else:
        print(
            f"every combination is in the {args.variant} train split: no held-out rows\n"
        )

    print("samples along each label axis (is a gap the digit, the fg, or the bg?)")
    for axis in ("digit", "fg", "bg"):
        print(samples.groupby(axis)[IMAGE_COLUMNS].mean().T, "\n")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        images.to_csv(args.output / "images.csv", index=False)
        evaluation.spread.to_csv(args.output / "spread.csv", index=False)
        evaluation.density.to_csv(args.output / "density.csv", index=False)
        print(f"wrote frames to {args.output}")


if __name__ == "__main__":
    main()

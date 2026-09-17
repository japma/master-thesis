"""Stage 1: write a run of latents and labels.

Checkpoints are always wandb artifacts, `name` or `name:version`.

    uv run generate_run real --variant uniform --split test --vae variational_colour_mnist_uniform:v1
    uv run generate_run model --model-type cspn --checkpoint psinet_colour_mnist_uniform:v1 \\
        --model-name cspn_std0.6 --std-correction 0.6 --seed 0 \\
        --vae variational_colour_mnist_uniform:v1 \\
        --reference eval_runs/colour_mnist_uniform_test__real__seed0
"""

import argparse
from pathlib import Path

from evaluation.generate import generate_model_run, generate_real_run
from evaluation.models import MODEL_TYPES
from utils.reproducibility import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=Path("eval_runs"))
    parser.add_argument("--batch-size", type=int, default=256)
    kinds = parser.add_subparsers(dest="kind", required=True)

    real = kinds.add_parser("real", help="encode a real split through the VAE")
    real.add_argument("--variant", default="uniform")
    real.add_argument("--split", default="test")
    real.add_argument(
        "--vae", required=True, help="wandb artifact, name or name:version"
    )
    real.add_argument("--data-root", type=Path, default=Path("data"))

    model = kinds.add_parser("model", help="sample a conditional model")
    model.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    model.add_argument(
        "--checkpoint", required=True, help="wandb artifact, name or name:version"
    )
    model.add_argument(
        "--model-name", required=True, help="label for this run, e.g. cspn_std0.6"
    )
    model.add_argument(
        "--vae", required=True, help="wandb artifact, name or name:version"
    )
    model.add_argument(
        "--reference", type=Path, required=True, help="real run to take labels from"
    )
    model.add_argument("--seed", type=int, default=0)
    model.add_argument("--std-correction", type=float, default=1.0)
    model.add_argument("--n-per-label", type=int, default=1)

    args = parser.parse_args()
    device = resolve_device()

    if args.kind == "real":
        run_dir = generate_real_run(
            variant=args.variant,
            split=args.split,
            vae_artifact=args.vae,
            output_root=args.output_root,
            device=device,
            data_root=args.data_root,
            batch_size=args.batch_size,
        )
    else:
        run_dir = generate_model_run(
            model_type=args.model_type,
            model_artifact=args.checkpoint,
            model_name=args.model_name,
            vae_artifact=args.vae,
            reference_dir=args.reference,
            output_root=args.output_root,
            seed=args.seed,
            std_correction=args.std_correction,
            device=device,
            n_per_label=args.n_per_label,
            batch_size=args.batch_size,
        )
    print(run_dir)


if __name__ == "__main__":
    main()

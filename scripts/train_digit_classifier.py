"""Train the fixed digit classifier the generation metrics judge samples with.

    uv run train_digit_classifier                 # 3 epochs on colour_mnist_uniform
    uv run train_digit_classifier --epochs 5
"""

import argparse

from dataset_loaders import build_data_loaders
from evaluation.classifier import (
    CLASSIFIER_PATH,
    save_digit_classifier,
    train_digit_classifier,
)
from utils.config import DatasetConfig
from utils.reproducibility import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="colour_mnist_uniform")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device()

    train_loader, val_loader = build_data_loaders(
        DatasetConfig(
            name=args.dataset, channels=3, height=28, width=28, num_classes=10
        ),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(f"Training digit classifier on {args.dataset} | device={device}")
    model, accuracy = train_digit_classifier(
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    if accuracy < 0.95:
        print(
            f"WARNING: validation accuracy {accuracy:.3f} is low for a judge -- digit "
            "metrics computed with it will understate every model. Train longer."
        )
    save_digit_classifier(model, CLASSIFIER_PATH)


if __name__ == "__main__":
    main()

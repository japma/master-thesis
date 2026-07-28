import csv
import os

import torch
from torchvision.utils import save_image

from dataset_loaders import build_data_loaders
from utils.config import DatasetConfig

_FG_COLOURS = {
    "red": (1.0, 0, 0),
    "green": (0, 1.0, 0),
    "blue": (0, 0, 1.0),
}

_BG_COLOURS = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}


def _label_to_colour_names(labels: torch.Tensor) -> tuple[list[str], list[str]]:
    fg_names = []
    bg_names = []
    for label in labels:
        if label == 0:
            fg_names.append("red")
            bg_names.append("white")
        elif label == 1:
            fg_names.append("green")
            bg_names.append("black")
        elif label == 2:
            fg_names.append("blue")
            bg_names.append("white")
        else:
            fg_names.append("red")
            bg_names.append("black")
    return fg_names, bg_names


def _colour_tensor(names: list[str]) -> torch.Tensor:
    """(B, 3) tensor of RGB values in [0, 1], one row per sample."""
    vals = torch.tensor(
        [_FG_COLOURS[n] if n in _FG_COLOURS else _BG_COLOURS[n] for n in names]
    )
    return vals


def _tint_batch(imgs: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    fg = fg.view(-1, 3, 1, 1)
    bg = bg.view(-1, 3, 1, 1)
    alpha = imgs
    return alpha * fg + (1 - alpha) * bg


def _process_split(loader, split_name: str, root: str) -> None:
    img_dir = os.path.join(root, split_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(root, split_name, "labels.csv")

    idx = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "fg_colour", "bg_colour"])

        for images, labels in loader:
            fg_names, bg_names = _label_to_colour_names(labels)
            fg = _colour_tensor(fg_names)
            bg = _colour_tensor(bg_names)

            tinted = _tint_batch(images, fg, bg)

            for i in range(tinted.size(0)):
                filename = f"{idx:06d}.png"
                save_image(tinted[i], os.path.join(img_dir, filename))
                writer.writerow([idx, labels[i].item(), fg_names[i], bg_names[i]])
                idx += 1

    print(f"[{split_name}] saved {idx} images to {img_dir}")


def main() -> None:
    mnist_cfg = DatasetConfig(
        name="mnist", height=28, width=28, channels=1, num_classes=10
    )
    train_loader, test_loader = build_data_loaders(
        mnist_cfg,
        batch_size=32,
        num_workers=4,
        drop_last=False,
    )

    root = "data/colour-mnist"
    os.makedirs(root, exist_ok=True)

    _process_split(train_loader, "train", root)
    _process_split(test_loader, "test", root)


if __name__ == "__main__":
    main()

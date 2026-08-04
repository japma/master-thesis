import csv
import os
import random
from collections.abc import Iterable

import PIL.Image
import PIL.ImageOps
import torch
import tqdm
from torchvision.datasets import MNIST

_FG_COLOURS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "pink": (255, 165, 0),
}

_BG_COLOURS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
}


def _tint_image(
    img: PIL.Image.Image,
    fg_colour: tuple[int, int, int],
    bg_colour: tuple[int, int, int],
) -> PIL.Image.Image:
    img = img.convert("L")
    img = PIL.ImageOps.colorize(img, black=bg_colour, white=fg_colour)
    return img


def _label_to_colours(label: int) -> tuple[str, str]:
    match label:
        case 0:
            fg_options = ["red", "green", "blue"]
            return random.choice(fg_options), random.choice(list(_BG_COLOURS.keys()))
        case 1:
            return random.choice(list(_FG_COLOURS.keys())), "black"
        case 2:
            return random.choice(list(_FG_COLOURS.keys())), "white"
        case _:
            return random.choice(list(_FG_COLOURS.keys())), random.choice(
                list(_BG_COLOURS.keys())
            )


def _process_split(
    dataset: torch.utils.data.Dataset, split_name: str, root: str
) -> None:
    img_dir = os.path.join(root, split_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(root, split_name, "labels.csv")

    idx = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "fg_colour", "bg_colour", "filename"])

        assert isinstance(dataset, Iterable)

        for image, label in tqdm.tqdm(dataset):
            fg_colour, bg_colour = _label_to_colours(label)
            tinted = _tint_image(image, _FG_COLOURS[fg_colour], _BG_COLOURS[bg_colour])

            filename = f"{idx:06d}.png"
            tinted.save(os.path.join(img_dir, filename))

            writer.writerow([idx, label, fg_colour, bg_colour, filename])
            idx += 1

    print(f"[{split_name}] saved {idx} images to {img_dir}")


def main() -> None:
    train_ds = MNIST(root="./data", train=True, download=True)
    test_ds = MNIST(root="./data", train=False, download=True)

    root = "data/colour-mnist"
    os.makedirs(root, exist_ok=True)

    _process_split(train_ds, "train", root)
    _process_split(test_ds, "test", root)


if __name__ == "__main__":
    main()

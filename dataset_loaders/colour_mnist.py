from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import datasets


def _tint_image(
    img: Image.Image,
    fg_colour: tuple[int, int, int],
    bg_colour: tuple[int, int, int],
) -> Image.Image:
    tinted = ImageOps.colorize(img, black=bg_colour, white=fg_colour)
    return tinted


def _extend_label_with_colours(label: int) -> tuple[int, int, int]:
    if label == 0:
        return 0, 0, 0
    elif label == 1:
        return 1, 1, 1
    elif label == 2:
        return 2, 2, 0
    else:
        return label, 0, 1


class ColourMNIST(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        download: bool = True,
    ):
        self.root = Path(root)
        self.train = train
        self.download = download
        self.transform = transform
        self.dataset = datasets.MNIST(
            root=self.root,
            train=train,
            transform=None,
            download=download,
        )

        self._FG_COLOURS = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        ]

        self._BG_COLOURS = [
            (255, 255, 255),  # White
            (0, 0, 0),  # Black
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        img, digit_label = self.dataset[index]

        label = torch.tensor(_extend_label_with_colours(digit_label))

        tinted_img = _tint_image(
            img,
            fg_colour=self._FG_COLOURS[label[1]],
            bg_colour=self._BG_COLOURS[label[2]],
        )

        if self.transform is not None:
            tinted_img = self.transform(tinted_img)

        return tinted_img, digit_label

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# TODO consolidate into one file with the generation script
FG_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "cyan": 3,
    "yellow": 4,
    "pink": 5,
}

BG_TO_IDX = {
    "white": 0,
    "black": 1,
    "grey": 2,
}


class ColourMNIST(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
    ):
        self.root = Path(root) / "colour-mnist"
        self.transform = transform
        self.split = "train" if train else "test"
        self.csv_path = self.root / self.split / "labels.csv"

        self.csv_reader = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.csv_reader)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        row = self.csv_reader.iloc[index]
        img_path = str(self.root / self.split / "images" / row["filename"])
        img = Image.open(img_path)
        target = torch.tensor(
            [row["label"], FG_TO_IDX[row["fg_colour"]], BG_TO_IDX[row["bg_colour"]]]
        )

        if self.transform:
            img = self.transform(img)

        return img, target

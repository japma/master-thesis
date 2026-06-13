from pathlib import Path
from typing import Optional, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class Cub200(Dataset):
    def __init__(
        self, root: str | Path, train: bool = True, transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform

        self.class_names = self._load_class_names()

        self.samples = self._load_samples()

    def _load_class_names(self) -> list[tuple[int, str]]:
        class_name_path = self.root / "classes.txt"
        class_names = []
        if not class_name_path.exists():
            raise FileNotFoundError(f"{class_name_path} does not exist")

        with open(class_name_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_names.append((int(parts[0]), parts[1]))

        return class_names

    def _load_samples(self) -> list[tuple[int, Path]]:
        train_ids = self._load_split_ids()
        image_class_path = self.root / "image_class_labels.txt"
        if not image_class_path.exists():
            raise FileNotFoundError(f"{image_class_path} does not exist")

        images_path = self.root / "images"
        images_desc_path = self.root / "images.txt"
        if not images_desc_path.exists():
            raise FileNotFoundError(f"{images_desc_path} does not exist")

        with open(image_class_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()

    def _load_split_ids(self) -> list[int]:
        tt_split_path = self.root / "train_test_split.txt"
        if not tt_split_path.exists():
            raise FileNotFoundError(f"{tt_split_path} does not exist")

        tt_split_ids = []
        with open(tt_split_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if bool(parts[1]) == self.train:
                    tt_split_ids.append(int(parts[0]))

        return tt_split_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

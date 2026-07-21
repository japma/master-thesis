from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class Cub200Dataset(Dataset):
    def __init__(
        self, root: str | Path, train: bool = True, transform: Callable | None = None
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform

        self.class_names = self._load_class_names()

        self.samples = self._load_samples()

    def _load_class_names(self) -> dict[int, str]:
        class_name_path = self.root / "classes.txt"
        if not class_name_path.exists():
            raise FileNotFoundError(f"{class_name_path} does not exist")

        with open(class_name_path) as f:
            return {
                int(parts[0]): parts[1] for line in f if (parts := line.strip().split())
            }

    def _load_samples(self) -> list[tuple[int, Path]]:
        train_ids = set(self._load_split_ids())

        image_class_path = self.root / "image_class_labels.txt"
        if not image_class_path.exists():
            raise FileNotFoundError(f"{image_class_path} does not exist")

        id_to_label = {}
        with open(image_class_path) as f:
            for line in f:
                parts = line.strip().split()
                id_to_label[int(parts[0])] = int(parts[1]) - 1

        images_desc_path = self.root / "images.txt"
        if not images_desc_path.exists():
            raise FileNotFoundError(f"{images_desc_path} does not exist")

        samples = []
        with open(images_desc_path) as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                if img_id in train_ids:
                    img_path = self.root / "images" / parts[1]
                    samples.append((id_to_label[img_id], img_path))

        return samples

    def _load_split_ids(self) -> list[int]:
        tt_split_path = self.root / "train_test_split.txt"
        if not tt_split_path.exists():
            raise FileNotFoundError(f"{tt_split_path} does not exist")

        tt_split_ids = []
        with open(tt_split_path) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if (parts[1] == "1") == self.train:
                    tt_split_ids.append(int(parts[0]))

        return tt_split_ids

    def download(self) -> None:
        raise NotImplementedError(
            "Automatic downloading is not implemented for CUB-200-2011."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label, image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

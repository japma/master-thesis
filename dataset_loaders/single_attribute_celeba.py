from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class SingleAttributeCelebA(Dataset):
    def __init__(
        self,
        attribute: str,
        root: Path,
        split: str,
        download: bool = True,
        transform: Callable | None = None,
    ) -> None:
        self.dataset = datasets.CelebA(
            root=root, split=split, download=download, transform=transform
        )
        # print(self.dataset.attr_names)
        if attribute not in self.dataset.attr_names:
            raise ValueError(f"Attribute '{attribute}' not found in CelebA dataset.")
        self.attribute_index = self.dataset.attr_names.index(attribute)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.dataset[index]
        attribute_value = self.dataset.attr[index][self.attribute_index]
        return image, attribute_value

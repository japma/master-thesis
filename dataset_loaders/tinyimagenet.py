"""Tiny ImageNet dataset utilities."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    """Local Tiny ImageNet dataset.

    Expected structure is the standard tiny-imagenet-200 folder layout.
    """

    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(
                f"Missing wnids.txt at {wnids_path}. "
                "Expected Tiny ImageNet root directory."
            )

        with wnids_path.open("r", encoding="utf-8") as file:
            self.classes = [line.strip() for line in file if line.strip()]

        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []

        if split == "train":
            self._load_train_samples()
        elif split == "val":
            self._load_val_samples()
        else:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for Tiny ImageNet split '{split}' in {self.root}."
            )

    def _load_train_samples(self):
        train_dir = self.root / "train"
        for class_name in self.classes:
            image_dir = train_dir / class_name / "images"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.JPEG")):
                self.samples.append((image_path, self.class_to_idx[class_name]))

    def _load_val_samples(self):
        val_dir = self.root / "val"
        annotations_path = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"

        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Missing val_annotations.txt at {annotations_path}."
            )

        with annotations_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                image_path = images_dir / image_name
                if image_path.exists():
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

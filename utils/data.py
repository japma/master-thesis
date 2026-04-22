"""Dataset and data loader helpers."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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


def get_data_loaders(dataset_name, batch_size=32, dataset_kwargs=None):
    """Load dataset with data loaders.

    Args:
        dataset_name: Name of the dataset to load. Must match a class in torchvision.datasets.
        batch_size: Batch size for training and validation.
        dataset_kwargs: Optional kwargs passed to torchvision dataset constructors
            (e.g., {"split": "letters"} for EMNIST).

    Returns:
        Tuple of (train_loader, test_loader).
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_kwargs = dict(dataset_kwargs or {})

    if dataset_name == "TinyImageNet":
        tiny_root = dataset_kwargs.pop("root", "./data/tiny-imagenet-200")
        tiny_root = dataset_kwargs.pop("data_dir", tiny_root)
        image_size = dataset_kwargs.pop("image_size", 64)

        if len(dataset_kwargs) > 0:
            unknown_keys = ", ".join(sorted(dataset_kwargs.keys()))
            raise ValueError(
                f"Unsupported dataset_kwargs for TinyImageNet: {unknown_keys}"
            )

        tiny_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = TinyImageNetDataset(
            root=tiny_root,
            split="train",
            transform=tiny_transform,
        )
        test_dataset = TinyImageNetDataset(
            root=tiny_root,
            split="val",
            transform=tiny_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    # Remap to [0, 25] for EMNIST
    if dataset_name == "EMNIST" and dataset_kwargs.get("split") == "letters":
        existing_target_transform = dataset_kwargs.get("target_transform")

        def _emnist_letters_to_zero_based(label):
            label = label - 1
            if existing_target_transform is not None:
                return existing_target_transform(label)
            return label

        dataset_kwargs["target_transform"] = _emnist_letters_to_zero_based

    dataset_class = getattr(datasets, dataset_name)
    train_dataset = dataset_class(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        **dataset_kwargs,
    )
    test_dataset = dataset_class(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        **dataset_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

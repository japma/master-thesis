"""Dataset and data loader helpers."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .tinyimagenet import TinyImageNetDataset


def _default_transform():
    return transforms.Compose([transforms.ToTensor()])


def _build_tinyimagenet_loaders(batch_size, dataset_kwargs):
    tiny_root = dataset_kwargs.pop("root", "./data/tiny-imagenet-200")
    tiny_root = dataset_kwargs.pop("data_dir", tiny_root)
    image_size = dataset_kwargs.pop("image_size", 64)

    if len(dataset_kwargs) > 0:
        unknown_keys = ", ".join(sorted(dataset_kwargs.keys()))
        raise ValueError(f"Unsupported dataset_kwargs for TinyImageNet: {unknown_keys}")

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


def _build_torchvision_loaders(dataset_name, batch_size, dataset_kwargs):
    # Remap to [0, 25] for EMNIST letters split.
    if dataset_name == "EMNIST" and dataset_kwargs.get("split") == "letters":
        existing_target_transform = dataset_kwargs.get("target_transform")

        def _emnist_letters_to_zero_based(label):
            label = label - 1
            if existing_target_transform is not None:
                return existing_target_transform(label)
            return label

        dataset_kwargs["target_transform"] = _emnist_letters_to_zero_based

    dataset_class = getattr(datasets, dataset_name)
    common_kwargs = {
        "root": "./data",
        "download": True,
        "transform": _default_transform(),
        **dataset_kwargs,
    }

    train_dataset = dataset_class(train=True, **common_kwargs)
    test_dataset = dataset_class(train=False, **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_data_loaders(dataset_name, batch_size=32, dataset_kwargs=None):
    """Load dataset with data loaders.

    Args:
        dataset_name: Name of the dataset to load. Must match a class in
            torchvision.datasets or be "TinyImageNet".
        batch_size: Batch size for training and validation.
        dataset_kwargs: Optional kwargs passed to dataset constructors.

    Returns:
        Tuple of (train_loader, test_loader).
    """

    dataset_kwargs = dict(dataset_kwargs or {})

    if dataset_name == "TinyImageNet":
        return _build_tinyimagenet_loaders(batch_size, dataset_kwargs)

    return _build_torchvision_loaders(dataset_name, batch_size, dataset_kwargs)

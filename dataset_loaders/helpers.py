import os
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.celeba import CelebA
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.flowers102 import Flowers102
from torchvision.datasets.mnist import MNIST, FashionMNIST

from dataset_loaders.binarymnist import BinaryMNISTDataset
from dataset_loaders.colour_mnist import DEFAULT_VARIANT, ColourMNIST
from dataset_loaders.cub200 import Cub200Dataset
from dataset_loaders.single_attribute_celeba import SingleAttributeCelebA
from dataset_loaders.tinyimagenet import TinyImageNetDataset
from utils.config import DatasetConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(3, 1, 1)


def _load_mnist(train: bool = True, size: tuple[int, int] = (28, 28)) -> MNIST:
    return datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_grayscale_to_rgb),
            ]
        ),
    )


def _load_binary_mnist(
    train: bool = True, size: tuple[int, int] = (28, 28)
) -> BinaryMNISTDataset:
    return BinaryMNISTDataset(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_grayscale_to_rgb),
            ]
        ),
    )


def _load_cifar10(train: bool = True, size: tuple[int, int] = (128, 128)) -> CIFAR10:
    return datasets.CIFAR10(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_fashion_mnist(
    train: bool = True, size: tuple[int, int] = (128, 128)
) -> FashionMNIST:
    return datasets.FashionMNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_grayscale_to_rgb),
            ]
        ),
    )


# TODO fix the path here
def _load_tinyimagenet(
    train: bool = True, size: tuple[int, int] = (128, 128)
) -> TinyImageNetDataset:
    split = "train" if train else "val"
    return TinyImageNetDataset(
        root="./data/tiny-imagenet-200",
        split=split,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_flowers102(
    train: bool = True, size: tuple[int, int] = (128, 128)
) -> Flowers102:
    train_transform = transforms.Compose(
        [
            # transforms.RandomRotation(10, expand=True),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    # Intentionally inverted so that we use the larger split for training
    return datasets.Flowers102(
        root=DATA_DIR,
        split="test" if train else "train",
        download=True,
        transform=train_transform if train else test_transform,
    )


def _load_cub200(
    train: bool = True, size: tuple[int, int] = (128, 128)
) -> Cub200Dataset:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10, expand=True),
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    return Cub200Dataset(
        root=os.path.join(DATA_DIR, "CUB_200_2011/CUB_200_2011/"),
        train=train,
        transform=train_transform if train else test_transform,
    )


def _load_celeba(train: bool = True, size: tuple[int, int] = (64, 64)) -> CelebA:
    return datasets.CelebA(
        root=DATA_DIR,
        split="train" if train else "test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        ),
    )


def _load_celeba_single_attribute(
    train: bool = True, size: tuple[int, int] = (64, 64)
) -> SingleAttributeCelebA:
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    attribute = "Male"
    return SingleAttributeCelebA(
        attribute=attribute,
        root=DATA_DIR,
        split="train" if train else "test",
        download=True,
        transform=transform,
    )


def _load_colour_mnist(
    train: bool = True,
    size: tuple[int, int] = (28, 28),
    variant: str = DEFAULT_VARIANT,
) -> ColourMNIST:
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    # `train=False` is the training loop's *validation* loader, so it maps to the `val`
    # split rather than `test` — test stays untouched until final numbers. The `test` and
    # `*_full` splits are constructed directly by eval code via `ColourMNIST(split=...)`.
    return ColourMNIST(
        root=DATA_DIR,
        split="train" if train else "val",
        variant=variant,
        transform=transform,
    )


_DATASETS = {
    "mnist": _load_mnist,
    "binary_mnist": _load_binary_mnist,
    "fashion_mnist": _load_fashion_mnist,
    "cifar10": _load_cifar10,
    "tinyimagenet": _load_tinyimagenet,
    "flowers102": _load_flowers102,
    "cub200": _load_cub200,
    "celeba": _load_celeba,
    "celeba_single_attribute": _load_celeba_single_attribute,
    # One entry per generated colour-MNIST variant; add a row for each new weight table.
    "colour_mnist_uniform": partial(_load_colour_mnist, variant="uniform"),
    "colour_mnist_skewed": partial(_load_colour_mnist, variant="skewed"),
    "colour_mnist_rgb": partial(_load_colour_mnist, variant="rgb"),
}


def build_data_loaders(
    cfg: DatasetConfig,
    batch_size: int = 32,
    shuffle_test: bool = False,
    num_workers: int = 8,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
    loader_fn = _DATASETS.get(cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")

    size = (cfg.height, cfg.width)

    train_dataset = loader_fn(train=True, size=size)
    test_dataset = loader_fn(train=False, size=size)

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        shuffle=shuffle_test,
    )

    return train_loader, test_loader


def download_datasets() -> None:
    """Download all available datasets by instantiating all of them"""
    for name in _DATASETS:
        ds_fn = _DATASETS[name]
        print(f"Processing dataset '{name}' ...")
        ds_fn(train=True)

"""Dataset and dataset loader helpers."""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .tinyimagenet import TinyImageNetDataset


def _load_mnist(train=True):
    return datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_cifar10(train=True):
    return datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_fashion_mnist(train=True):
    return datasets.FashionMNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_tinyimagenet(split="train"):
    return TinyImageNetDataset(
        root="./data/tiny-imagenet-200",
        split=split,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


_DATASETS = {
    "MNIST": _load_mnist,
    "FashionMNIST": _load_fashion_mnist,
    "CIFAR10": _load_cifar10,
    "TinyImageNet": _load_tinyimagenet,
}


def _build_data_sets(cfg):
    loader_fn = _DATASETS.get(cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")

    # TinyImageNet uses 'split' parameter instead of 'train'
    if cfg.name == "TinyImageNet":
        return loader_fn(split="train"), loader_fn(split="val")
    else:
        return loader_fn(train=True), loader_fn(train=False)


def get_data_loaders(cfg, batch_size=32):
    train_dataset, test_dataset = _build_data_sets(cfg)

    max_workers = os.cpu_count()
    if max_workers is None:
        max_workers = 8
    else:
        max_workers = max_workers // 2

    train_loader = DataLoader(
        train_dataset,
        num_workers=max_workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=max_workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader

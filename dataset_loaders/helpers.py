"""Dataset and dataset loader helpers."""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .binarymnist import BinaryMNISTDataset
from .tinyimagenet import TinyImageNetDataset
from .coco import CocoDataset


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


def _load_binary_mnist(train=True):
    return BinaryMNISTDataset(
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


def _load_tinyimagenet(train=True):
    if train:
        split = "train"
    else:
        split = "val"
    return TinyImageNetDataset(
        root="./data/tiny-imagenet-200",
        split=split,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


def _load_coco(train=True):
    root = f"./data/coco-2017/{'train' if train else 'val'}/"
    ann_file = (
        f"./data/coco-2017/annotations/instances_{'train' if train else 'val'}2017.json"
    )
    return CocoDataset(
        root=root,
        ann_file=ann_file,
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        ),
    )


_DATASETS = {
    "MNIST": _load_mnist,
    "BinaryMNIST": _load_binary_mnist,
    "FashionMNIST": _load_fashion_mnist,
    "CIFAR10": _load_cifar10,
    "TinyImageNet": _load_tinyimagenet,
    "COCO": _load_coco,
}


def build_data_loaders(cfg, batch_size=32) -> tuple[DataLoader, DataLoader]:
    loader_fn = _DATASETS.get(cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")
    train_dataset = loader_fn(train=True)
    test_dataset = loader_fn(train=False)

    max_workers = os.cpu_count()
    if max_workers is None:
        max_workers = 8
    else:
        max_workers = max_workers // 2

    train_loader = DataLoader(
        train_dataset,
        num_workers=max_workers,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=max_workers,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        drop_last=True,
    )

    return train_loader, test_loader

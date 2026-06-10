import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import DatasetConfig
from .binarymnist import BinaryMNISTDataset
from .tinyimagenet import TinyImageNetDataset
from .coco import CocoDataset, CocoCachedDataset


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


def _load_flowers102(train=True):
    return datasets.Flowers102(
        root="./data",
        split="train" if train else "test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        ),
    )


def _load_celeba(train=True):
    raise NotImplementedError


_DATASETS = {
    "mnist": _load_mnist,
    "binary_mnist": _load_binary_mnist,
    "fashion_mnist": _load_fashion_mnist,
    "cifar10": _load_cifar10,
    "tinyimagenet": _load_tinyimagenet,
    "coco": _load_coco,
    "flowers102": _load_flowers102,
    "celeba": _load_celeba,
}


def build_data_loaders(
    cfg: DatasetConfig, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    loader_fn = _DATASETS.get(cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")
    train_dataset = loader_fn(train=True)
    test_dataset = loader_fn(train=False)

    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=cfg.num_workers,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        persistent_workers=True,
        drop_last=True,
    )

    return train_loader, test_loader

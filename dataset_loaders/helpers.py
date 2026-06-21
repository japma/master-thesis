import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_loaders.single_class_sampler import SingleClassSampler
from utils.config import DatasetConfig
from dataset_loaders.binarymnist import BinaryMNISTDataset
from dataset_loaders.tinyimagenet import TinyImageNetDataset
from dataset_loaders.coco import CocoDataset, CocoCachedDataset
from dataset_loaders.cub200 import Cub200Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(3, 1, 1)


def _load_mnist(train=True, size: tuple[int, int] = (28, 28)):
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


def _load_binary_mnist(train=True, size: tuple[int, int] = (28, 28)):
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


def _load_cifar10(train=True, size: tuple[int, int] = (128, 128)):
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


def _load_fashion_mnist(train=True, size: tuple[int, int] = (128, 128)):
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


def _load_tinyimagenet(train=True, size: tuple[int, int] = (128, 128)):
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


def _load_coco(train=True, size: tuple[int, int] = (128, 128)):
    root = f"./data/coco-2017/{'train' if train else 'val'}/"
    ann_file = (
        f"./data/coco-2017/annotations/instances_{'train' if train else 'val'}2017.json"
    )
    return CocoDataset(
        root=root,
        ann_file=ann_file,
        transform=transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        ),
    )


def _load_flowers102(train=True, size: tuple[int, int] = (128, 128)):
    return datasets.Flowers102(
        root=DATA_DIR,
        split="test" if train else "train",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        ),
    )


def _load_cub200(train=True, size: tuple[int, int] = (128, 128)):
    return Cub200Dataset(
        root=os.path.join(DATA_DIR, "CUB_200_2011/CUB_200_2011/"),
        train=train,
        transform=transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        ),
    )


def _load_celeba(train=True, size: tuple[int, int] = (64, 64)):
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


_DATASETS = {
    "mnist": _load_mnist,
    "binary_mnist": _load_binary_mnist,
    "fashion_mnist": _load_fashion_mnist,
    "cifar10": _load_cifar10,
    "tinyimagenet": _load_tinyimagenet,
    "coco": _load_coco,
    "flowers102": _load_flowers102,
    "cub200": _load_cub200,
    "celeba": _load_celeba,
}


def build_data_loaders(
    cfg: DatasetConfig, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    loader_fn = _DATASETS.get(cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")

    size = (cfg.height, cfg.width)

    train_dataset = loader_fn(train=True, size=size)
    test_dataset = loader_fn(train=False, size=size)

    train_sampler = SingleClassSampler(train_dataset, batch_size)
    test_sampler = SingleClassSampler(test_dataset, batch_size)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        persistent_workers=True,
        drop_last=True,
    )

    return train_loader, test_loader

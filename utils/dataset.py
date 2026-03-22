"""
utils/dataset.py
CIFAR-10 dataset loading and preprocessing utilities.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from parameters import DataConfig


def get_cifar10_transforms(resize: bool = False, image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and test transforms for CIFAR-10.

    Args:
        resize: If True, resize images to image_size (for ImageNet-pretrained backbones).
        image_size: Target image size when resize=True.

    Returns:
        Tuple of (train_transform, test_transform).
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    if resize:
        base_train = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        base_test = [transforms.Resize(image_size)]
    else:
        base_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        base_test = []

    train_transform = transforms.Compose(base_train + [transforms.ToTensor(), normalize])
    test_transform = transforms.Compose(base_test + [transforms.ToTensor(), normalize])

    return train_transform, test_transform


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test DataLoaders.

    Args:
        cfg: DataConfig instance with dataset parameters.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_transform, test_transform = get_cifar10_transforms(cfg.resize, cfg.image_size)

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader, test_loader

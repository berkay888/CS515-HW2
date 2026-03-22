"""
models/resnet.py
ResNet-18 wrappers for Transfer Learning (Part A) and scratch training (Part B).
"""

import torch
import torch.nn as nn
from torchvision import models


def get_resnet18_transfer(num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained ResNet-18 and adapt it for CIFAR-10.

    Option 1 (freeze_backbone=True):
        - Input images are expected to be resized to 224x224.
        - All layers except the final FC are frozen.
        - Only the FC layer is trained.

    Option 2 (freeze_backbone=False):
        - Input images remain 32x32.
        - The first conv layer is replaced with a smaller kernel.
        - All layers are fine-tuned.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze early layers (Option 1). Else fine-tune all (Option 2).
        pretrained: Load ImageNet pretrained weights.

    Returns:
        Modified ResNet-18 model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Option 1: freeze everything except FC
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Option 2: replace first conv to handle 32x32 input
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # remove maxpool to preserve spatial size

    # Replace final FC for CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_resnet18_scratch(num_classes: int = 10) -> nn.Module:
    """
    Build a ResNet-18 adapted for CIFAR-10, trained from scratch.

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNet-18 model without pretrained weights, modified for 32x32 inputs.
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

"""
models/mobilenet.py
MobileNetV2 adapted for CIFAR-10 classification (student model in Part B soft-label KD).
"""

import torch
import torch.nn as nn
from torchvision import models


def get_mobilenetv2(num_classes: int = 10) -> nn.Module:
    """
    Build a MobileNetV2 adapted for CIFAR-10 (32x32 inputs), trained from scratch.

    Modifications:
        - First conv stride reduced from 2 to 1 to preserve spatial resolution.
        - Classifier head replaced for CIFAR-10.

    Args:
        num_classes: Number of output classes.

    Returns:
        MobileNetV2 model.
    """
    model = models.mobilenet_v2(weights=None)

    # Adapt for 32x32 input: reduce stride in first conv
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )

    return model

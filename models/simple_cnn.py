"""
models/simple_cnn.py
A lightweight CNN architecture for CIFAR-10 classification (student model).
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN with 3 convolutional blocks and 2 fully-connected layers.
    Used as the student model in knowledge distillation experiments.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

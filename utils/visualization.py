"""
utils/visualization.py
Training curves, confusion matrix, and t-SNE visualizations.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def plot_training_curves(
    train_losses: List[float],
    test_losses: List[float],
    train_accs: List[float],
    test_accs: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss and accuracy curves over epochs.

    Args:
        train_losses: List of training losses per epoch.
        test_losses: List of test losses per epoch.
        train_accs: List of training accuracies per epoch.
        test_accs: List of test accuracies per epoch.
        title: Plot title.
        save_path: If provided, saves the figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, test_losses, label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, test_accs, label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tsne(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    title: str = "t-SNE",
    save_path: Optional[str] = None,
    max_samples: int = 2000,
) -> None:
    """
    Extract penultimate-layer features and plot t-SNE.

    Args:
        model: Trained model with a forward_features() method or standard forward().
        loader: DataLoader to extract features from.
        device: Torch device.
        title: Plot title.
        save_path: If provided, saves the figure to this path.
        max_samples: Maximum number of samples to use for t-SNE.
    """
    from sklearn.manifold import TSNE

    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            out = model(images)
            features.append(out.cpu().numpy())
            labels.append(targets.numpy())
            if sum(len(f) for f in features) >= max_samples:
                break

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

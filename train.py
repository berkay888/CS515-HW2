"""
train.py
Training loop utilities used by main.py experiments.
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parameters import TrainConfig
from utils.metrics import accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int = 100,
    teacher: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model: Model being trained.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function. If teacher is provided, criterion should accept
                   (student_logits, teacher_logits, targets); otherwise (logits, targets).
        device: Torch device.
        log_interval: Log every N batches.
        teacher: Optional frozen teacher model for distillation.

    Returns:
        Tuple of (average_loss, top1_accuracy).
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)

        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = criterion(logits, teacher_logits, targets)
        else:
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        acc = accuracy(logits, targets, topk=(1,))[0]
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}  Acc@1: {acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Args:
        model: Model to evaluate.
        loader: DataLoader (typically test/val set).
        criterion: Standard CE loss for evaluation.
        device: Torch device.

    Returns:
        Tuple of (average_loss, top1_accuracy).
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = ce(logits, targets)
        acc = accuracy(logits, targets, topk=(1,))[0]
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    acc: float,
    save_dir: str,
    filename: str = "best.pth",
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        acc: Validation accuracy at this checkpoint.
        save_dir: Directory to save checkpoint.
        filename: Checkpoint filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(
        {"epoch": epoch, "state_dict": model.state_dict(),
         "optimizer": optimizer.state_dict(), "acc": acc},
        path,
    )
    print(f"  Checkpoint saved → {path}  (acc={acc:.2f}%)")

"""
utils/metrics.py
Accuracy computation and FLOPs counting utilities.
"""

from typing import Tuple
import torch
import torch.nn as nn


def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> list:
    """
    Compute top-k accuracy.

    Args:
        outputs: Model logits, shape (N, C).
        targets: Ground-truth labels, shape (N,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        List of top-k accuracy values (as percentages).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def count_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 32, 32)) -> str:
    """
    Count FLOPs and parameters using ptflops.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (C, H, W).

    Returns:
        Formatted string with GFLOPs and parameter count.
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        return f"MACs: {macs} | Params: {params}"
    except ImportError:
        return "ptflops not installed. Run: pip install ptflops"

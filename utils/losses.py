"""
utils/losses.py
Custom loss functions: Label Smoothing, Knowledge Distillation, Soft-Label KD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        num_classes: Number of output classes.
        smoothing: Smoothing factor epsilon (0 = standard CE).
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Raw model outputs, shape (N, C).
            targets: Ground-truth class indices, shape (N,).

        Returns:
            Scalar loss tensor.
        """
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class KnowledgeDistillationLoss(nn.Module):
    """
    Hinton-style Knowledge Distillation loss (KL-div + CE).

    Args:
        temperature: Distillation temperature T.
        alpha: Weight for distillation loss (1-alpha for CE loss).
        num_classes: Number of output classes.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, num_classes: int = 10) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined distillation + classification loss.

        Args:
            student_logits: Student model outputs, shape (N, C).
            teacher_logits: Teacher model outputs, shape (N, C).
            targets: Ground-truth labels, shape (N,).

        Returns:
            Scalar combined loss.
        """
        soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (self.T ** 2)
        ce_loss = self.ce(student_logits, targets)
        return self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss


class SoftLabelKDLoss(nn.Module):
    """
    Soft-label KD: teacher outputs assign probability to true class only;
    remaining probability is split equally among other classes.

    Args:
        temperature: Temperature for teacher softmax.
        alpha: Weight for soft-label loss.
        num_classes: Number of output classes.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, num_classes: int = 10) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft-label KD loss.

        Args:
            student_logits: Student model outputs, shape (N, C).
            teacher_logits: Teacher model outputs, shape (N, C).
            targets: Ground-truth labels, shape (N,).

        Returns:
            Scalar combined loss.
        """
        teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)

        # Build soft labels: p_true = teacher prob for true class, rest split equally
        N, C = student_logits.shape
        soft_labels = torch.full((N, C), fill_value=0.0, device=student_logits.device)
        true_probs = teacher_probs.gather(1, targets.unsqueeze(1))  # (N, 1)
        remainder = (1.0 - true_probs) / (C - 1)
        soft_labels = remainder.expand(N, C).clone()
        soft_labels.scatter_(1, targets.unsqueeze(1), true_probs)

        log_student = F.log_softmax(student_logits / self.T, dim=-1)
        distill_loss = F.kl_div(log_student, soft_labels, reduction="batchmean") * (self.T ** 2)
        ce_loss = self.ce(student_logits, targets)
        return self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss

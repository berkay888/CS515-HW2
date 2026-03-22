"""
main.py
Entry point for all CS515 HW1b experiments.

Experiments:
  transfer_opt1   — Transfer learning: resize + freeze backbone (Option 1)
  transfer_opt2   — Transfer learning: modify early layers + fine-tune (Option 2)
  scratch         — ResNet-18 from scratch without label smoothing
  scratch_ls      — ResNet-18 from scratch with label smoothing
  kd              — Knowledge distillation: SimpleCNN student + ResNet teacher
  soft_kd         — Soft-label KD: MobileNet student + ResNet teacher

Usage example:
  python main.py --experiment kd --epochs 50 --lr 0.01 --teacher_ckpt checkpoints/resnet_scratch_best.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from parameters import (
    DataConfig, TrainConfig,
    TransferLearningConfig, KnowledgeDistillationConfig,
    LabelSmoothingConfig, SoftLabelKDConfig,
)
from utils.dataset import get_dataloaders
from utils.losses import LabelSmoothingLoss, KnowledgeDistillationLoss, SoftLabelKDLoss
from utils.metrics import count_flops
from utils.visualization import plot_training_curves
from models import SimpleCNN, get_resnet18_transfer, get_resnet18_scratch, get_mobilenetv2
from train import train_one_epoch, evaluate, save_checkpoint


EXPERIMENTS = [
    "transfer_opt1", "transfer_opt2",
    "scratch", "scratch_ls",
    "kd", "soft_kd",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CS515 HW1b — Main experiment runner")
    parser.add_argument("--experiment", type=str, required=True, choices=EXPERIMENTS,
                        help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--teacher_ckpt", type=str, default="",
                        help="Path to teacher checkpoint (required for kd and soft_kd)")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon (used in scratch_ls)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature (used in kd and soft_kd)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Distillation loss weight (used in kd and soft_kd)")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "vgg16"],
                        help="Backbone for transfer learning experiments")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Dispatch and run the selected experiment."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    resize = args.experiment == "transfer_opt1"
    data_cfg = DataConfig(data_dir=args.data_dir, batch_size=args.batch_size, resize=resize)
    train_loader, test_loader = get_dataloaders(data_cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    teacher: nn.Module | None = None

    if args.experiment == "transfer_opt1":
        model = get_resnet18_transfer(freeze_backbone=True)
        ckpt_name = "transfer_opt1_best.pth"

    elif args.experiment == "transfer_opt2":
        model = get_resnet18_transfer(freeze_backbone=False)
        ckpt_name = "transfer_opt2_best.pth"

    elif args.experiment == "scratch":
        model = get_resnet18_scratch()
        ckpt_name = "resnet_scratch_best.pth"

    elif args.experiment == "scratch_ls":
        model = get_resnet18_scratch()
        ckpt_name = "resnet_scratch_ls_best.pth"

    elif args.experiment == "kd":
        assert args.teacher_ckpt, "--teacher_ckpt is required for kd experiment"
        model = SimpleCNN()
        teacher = get_resnet18_scratch()
        ckpt_data = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt_data["state_dict"])
        teacher = teacher.to(device)
        teacher.eval()
        ckpt_name = "simplecnn_kd_best.pth"

    elif args.experiment == "soft_kd":
        assert args.teacher_ckpt, "--teacher_ckpt is required for soft_kd experiment"
        model = get_mobilenetv2()
        teacher = get_resnet18_scratch()
        ckpt_data = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt_data["state_dict"])
        teacher = teacher.to(device)
        teacher.eval()
        ckpt_name = "mobilenet_softkd_best.pth"

    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.experiment == "scratch_ls":
        criterion = LabelSmoothingLoss(num_classes=10, smoothing=args.smoothing)
    elif args.experiment == "kd":
        criterion = KnowledgeDistillationLoss(temperature=args.temperature, alpha=args.alpha)
    elif args.experiment == "soft_kd":
        criterion = SoftLabelKDLoss(temperature=args.temperature, alpha=args.alpha)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    print(f"\n[{args.experiment}] Model complexity:")
    print(count_flops(model))
    if teacher is not None:
        print("Teacher complexity:")
        print(count_flops(teacher))

    # ── Training Loop ─────────────────────────────────────────────────────────
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, teacher=teacher
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_accs.append(tr_acc);   test_accs.append(te_acc)

        print(f"  Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%")
        print(f"  Test  Loss: {te_loss:.4f}  Test  Acc: {te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(model, optimizer, epoch, te_acc, args.save_dir, ckpt_name)

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs("results/figures", exist_ok=True)
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        title=args.experiment,
        save_path=f"results/figures/{args.experiment}_curves.png",
    )


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

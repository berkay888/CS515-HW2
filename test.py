"""
test.py
Load a saved checkpoint and evaluate it on the CIFAR-10 test set.
"""

import argparse
import torch

from parameters import DataConfig
from utils.dataset import get_dataloaders
from utils.metrics import accuracy, count_flops
from models import SimpleCNN, get_resnet18_scratch, get_mobilenetv2


MODEL_MAP = {
    "simple_cnn": SimpleCNN,
    "resnet18": get_resnet18_scratch,
    "mobilenet": get_mobilenetv2,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for test.py."""
    parser = argparse.ArgumentParser(description="CS515 HW1b — Test a saved model on CIFAR-10")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help="Model architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--flops", action="store_true", help="Print FLOPs / parameter count")
    return parser.parse_args()


def main() -> None:
    """Entry point for test.py."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    cfg = DataConfig(data_dir=args.data_dir, batch_size=args.batch_size)
    _, test_loader = get_dataloaders(cfg)

    # Build model & load weights
    model = MODEL_MAP[args.model]().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # FLOPs
    if args.flops:
        print(count_flops(model))

    # Evaluate
    total_acc, n_batches = 0.0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_acc += accuracy(logits, targets, topk=(1,))[0]
            n_batches += 1

    print(f"Test Accuracy: {total_acc / n_batches:.2f}%")


if __name__ == "__main__":
    main()

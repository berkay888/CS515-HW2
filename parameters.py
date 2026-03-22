"""
parameters.py
Dataclasses for all experiment hyperparameters and configurations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    resize: bool = False  # True => resize CIFAR-10 to 224x224 (Option 1), False => modify model (Option 2)
    image_size: int = 224  # used only if resize=True


@dataclass
class TrainConfig:
    """Configuration for training loop."""
    epochs: int = 50
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler_step: int = 20
    scheduler_gamma: float = 0.1
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    log_interval: int = 100


@dataclass
class TransferLearningConfig:
    """Configuration specific to Transfer Learning (Part A)."""
    backbone: str = "resnet18"          # "resnet18" or "vgg16"
    freeze_backbone: bool = True        # True => Option 1 (resize+freeze), False => Option 2 (modify+finetune)
    num_classes: int = 10
    pretrained: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class KnowledgeDistillationConfig:
    """Configuration for Knowledge Distillation (Part B)."""
    temperature: float = 4.0
    alpha: float = 0.7                  # Weight for distillation loss
    teacher_checkpoint: str = ""
    num_classes: int = 10
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class LabelSmoothingConfig:
    """Configuration for label smoothing experiments."""
    smoothing: float = 0.1
    num_classes: int = 10
    backbone: str = "resnet18"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class SoftLabelKDConfig:
    """Configuration for soft-label KD with MobileNet (Part B last step)."""
    teacher_checkpoint: str = ""
    num_classes: int = 10
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

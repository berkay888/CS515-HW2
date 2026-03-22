# CS515 HW1b — Transfer Learning & Knowledge Distillation

CIFAR-10 classification experiments using PyTorch.  
Covers Transfer Learning (Part A) and Knowledge Distillation (Part B).

---

## Project Structure

```
cs515-hw1b/
├── main.py              # Experiment runner (all 6 experiments)
├── train.py             # Training loop, evaluation, checkpoint saving
├── test.py              # Load checkpoint → evaluate on test set
├── parameters.py        # Dataclasses for all hyperparameters
├── requirements.txt
├── models/
│   ├── simple_cnn.py    # Lightweight student CNN
│   ├── resnet.py        # ResNet-18 (transfer + scratch)
│   └── mobilenet.py     # MobileNetV2 (soft-label KD student)
├── utils/
│   ├── dataset.py       # CIFAR-10 DataLoader factory
│   ├── losses.py        # LabelSmoothing / KD / SoftLabelKD losses
│   ├── metrics.py       # Accuracy, FLOPs counter
│   └── visualization.py # Training curves, t-SNE
└── results/
    └── figures/         # Saved plots
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Experiments

### Part A — Transfer Learning

**Option 1:** Resize CIFAR-10 → 224×224, freeze backbone, train FC only.
```bash
python main.py --experiment transfer_opt1 --epochs 30 --lr 0.01
```

**Option 2:** Modify early conv layers, fine-tune entire network on 32×32.
```bash
python main.py --experiment transfer_opt2 --epochs 50 --lr 0.01
```

---

### Part B — Knowledge Distillation

**Step 1:** Train ResNet-18 from scratch (no label smoothing).
```bash
python main.py --experiment scratch --epochs 100 --lr 0.1
```

**Step 2:** Train ResNet-18 from scratch with label smoothing.
```bash
python main.py --experiment scratch_ls --epochs 100 --lr 0.1 --smoothing 0.1
```

**Step 3:** Knowledge distillation — SimpleCNN student, ResNet teacher.
```bash
python main.py --experiment kd --epochs 50 --lr 0.01 \
  --teacher_ckpt checkpoints/resnet_scratch_best.pth \
  --temperature 4.0 --alpha 0.7
```

**Step 4:** Soft-label KD — MobileNetV2 student, ResNet teacher.
```bash
python main.py --experiment soft_kd --epochs 50 --lr 0.01 \
  --teacher_ckpt checkpoints/resnet_scratch_best.pth \
  --temperature 4.0 --alpha 0.7
```

---

### Evaluate a saved checkpoint

```bash
python test.py --model resnet18 --checkpoint checkpoints/resnet_scratch_best.pth --flops
```

---

## References

1. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.  
2. Hinton et al., "Distilling the Knowledge in a Neural Network," arXiv 2015.  
3. Müller et al., "When Does Label Smoothing Help?" NeurIPS 2019.  
4. Sandler et al., "MobileNetV2," CVPR 2018.  
5. Simonyan & Zisserman, "Very Deep Convolutional Networks," arXiv 2014.  
6. Szegedy et al., "Rethinking the Inception Architecture," CVPR 2016.  

"""
train_baseline_roi.py  —  Baseline CNN
====================================
Deliberately naive first attempt. Preserved as close to the original
as possible. The only changes from the original submission are:

  FIXED:  nn.Linear(128*32*32, 256) crashed at non-512 input sizes.
          Replaced with AdaptiveAvgPool2d(4,4) before flatten so the
          architecture is input-size agnostic. Linear(2048, 256) retained.

  ADDED:  EarlyStopping (patience=5, delta=1e-4) — Step 1 of brief.
  ADDED:  Model saving to Models/best_baseline.pth.
  ADDED:  Random seed for reproducibility.

Everything else is original: no normalisation, no batch norm, no
depthwise convs, no attention, basic Adam with fixed LR, rotation-only
augmentation. This intentional simplicity is what the improved and Lite
models are measured against.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stopping import EarlyStopping

#  Reproducibility 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths 
TRAIN = "Dataset/ROI/train"
VAL   = "Dataset/ROI/val"
TEST  = "Dataset/ROI/test"

os.makedirs("Models", exist_ok=True)
os.makedirs("Plots",  exist_ok=True)
CKPT = "Models/best_baseline_roi.pth"

# Hyper-parameters — original values preserved 
IMG_SIZE    = 224       # original — slow but kept for authenticity
BATCH       = 8         # original
NUM_EPOCHS  = 60        # raised from 10; ES fires before this anyway
LR          = 1e-3      # original
PATIENCE    = 5
DELTA       = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Transforms — original (no normalisation, rotation only) 
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),          # original augmentation
    transforms.ToTensor(),
    # no Normalize — this is intentionally omitted in the baseline
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Data 
train_ds = datasets.ImageFolder(TRAIN, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL,   transform=eval_tf)
test_ds  = datasets.ImageFolder(TEST,  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS)

print(f"Classes : {train_ds.classes}")
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}\n")

#  Model — original architecture with one structural fix 
class BrainTumorCNN(nn.Module):
    """
    Original four-block CNN.

    FIXED: The original code had nn.Linear(128 * 32 * 32, 256), which hardcodes
    a 512x512 input assumption in the linear layer dimensions and produces a
    ~134M parameter layer. Replaced with AdaptiveAvgPool2d(4,4) which gives a
    128x4x4=2048-d vector regardless of input size. Linear(2048, 256) kept.

    No batch norm, no residuals, no attention — intentionally basic.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,  32,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),   # FIX: replaces hardcoded spatial dims
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),   # 2048 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model     = BrainTumorCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()          # original — no label smoothing
optimizer = optim.Adam(model.parameters(), lr=LR)   # original

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Baseline CNN — {n_params:,} trainable parameters\n")

# Training helpers 
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# Training loop 
stopper = EarlyStopping(patience=PATIENCE, delta=DELTA,
                        checkpoint_path=CKPT, verbose=True, mode="min")
history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

hdr = f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  {'Va Loss':>9}  {'Va Acc':>8}  {'ES':>6}"
print(hdr); print("-" * len(hdr))

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch()
    vl_loss, vl_acc = evaluate(val_loader)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    print(f"{epoch:>6}  {tr_loss:>9.5f}  {tr_acc:>7.2f}%  "
          f"{vl_loss:>9.5f}  {vl_acc:>7.2f}%  {stopper.counter}/{PATIENCE}")

    if stopper.step(vl_loss, model):
        print(f"\n  Early stop at epoch {epoch}. "
              f"Best val loss {stopper.best_score:.5f} "
              f"(epoch {stopper.best_epoch + 1}).\n")
        break

stopper.load_best(model)

# Test 
test_loss, test_acc = evaluate(test_loader)
print(f"\nTest accuracy : {test_acc:.2f}%  |  Test loss : {test_loss:.5f}")

# Curves 
xs = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(xs, history["train_loss"], label="Train", color="#2196F3", lw=2)
ax1.plot(xs, history["val_loss"],   label="Val",   color="#F44336", lw=2)
ax1.axvline(stopper.best_epoch + 1, ls="--", color="#4CAF50", lw=1.5,
            label=f"Best (ep {stopper.best_epoch + 1})")
ax1.set_title("Baseline CNN — Loss"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(xs, history["train_acc"], label="Train", color="#2196F3", lw=2)
ax2.plot(xs, history["val_acc"],   label="Val",   color="#F44336", lw=2)
ax2.set_title("Baseline CNN — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Plots/baseline_curves_roi.png", dpi=150); plt.close()
print("Saved: Plots/baseline_curves_roi.png")
print(f"Best model: {CKPT}")

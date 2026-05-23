"""
train_improved_roi.py  —  Improved CNN
====================================
Step up from baseline:
  + BatchNorm after every conv block     (more stable training)
  + Higher resolution: 256x256          (vs 128 baseline)
  + Input normalisation                  (vs none in baseline)
  + Wider classifier head 128->512->128  (more capacity than original 128->64)
  + Cosine LR decay                      (vs fixed LR in baseline)
  + lr=3e-4                              (1e-3 destabilises BatchNorm)

Fixed random seed for reproducibility.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

from stopping import EarlyStopping

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN = "Dataset/ROI/train"
VAL   = "Dataset/ROI/val"
TEST  = "Dataset/ROI/test"

os.makedirs("Models", exist_ok=True)
os.makedirs("Plots",  exist_ok=True)
CKPT = "Models/best_improved_roi.pth"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH       = 16
NUM_EPOCHS  = 80
LR          = 3e-4      # FIXED: 1e-3 destabilises BatchNorm layers
PATIENCE    = 7         # slightly more tolerant given cosine LR valleys
DELTA       = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
# Normalisation added vs baseline — this is one of the explicit improvements
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Model ─────────────────────────────────────────────────────────────────────
class BetterBrainTumorCNN(nn.Module):
    """
    Improvements over baseline:
      - BatchNorm after every conv (stabilises gradients, allows lower LR)
      - Normalised inputs
      - Wider classifier head (512 hidden units vs baseline's 256)
      - No fixed spatial pooling — AdaptiveAvgPool2d(1,1) is resolution-agnostic

    Grad-CAM target: self.features[12]  (last Conv2d, 64->128 channels)
    Compatible with gradcam_3d.py.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  3, padding=1),   # [0]
            nn.BatchNorm2d(16), nn.ReLU(),        # [1][2]
            nn.MaxPool2d(2),                       # [3]

            nn.Conv2d(16,  32,  3, padding=1),   # [4]
            nn.BatchNorm2d(32), nn.ReLU(),        # [5][6]
            nn.MaxPool2d(2),                       # [7]

            nn.Conv2d(32,  64,  3, padding=1),   # [8]
            nn.BatchNorm2d(64), nn.ReLU(),        # [9][10]
            nn.MaxPool2d(2),                       # [11]

            nn.Conv2d(64,  128, 3, padding=1),   # [12] <- Grad-CAM target
            nn.BatchNorm2d(128), nn.ReLU(),       # [13][14]
            nn.MaxPool2d(2),                       # [15]

            nn.AdaptiveAvgPool2d((1, 1)),         # [16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model     = BetterBrainTumorCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Improved CNN — {n_params:,} trainable parameters\n")

# ── Training helpers ──────────────────────────────────────────────────────────
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


# ── Training loop ─────────────────────────────────────────────────────────────
stopper = EarlyStopping(patience=PATIENCE, delta=DELTA,
                        checkpoint_path=CKPT, verbose=True, mode="min")
history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

hdr = f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  {'Va Loss':>9}  {'Va Acc':>8}  {'LR':>9}  {'ES':>6}"
print(hdr); print("-" * len(hdr))

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch()
    vl_loss, vl_acc = evaluate(val_loader)
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    print(f"{epoch:>6}  {tr_loss:>9.5f}  {tr_acc:>7.2f}%  "
          f"{vl_loss:>9.5f}  {vl_acc:>7.2f}%  {lr:>9.2e}  "
          f"{stopper.counter}/{PATIENCE}")

    if stopper.step(vl_loss, model):
        print(f"\n  Early stop at epoch {epoch}. "
              f"Best val loss {stopper.best_score:.5f} "
              f"(epoch {stopper.best_epoch + 1}).\n")
        break

stopper.load_best(model)

# Mirror for gradcam_3d.py compatibility
shutil.copy(CKPT, "Models/brain_tumor_cnn_roi.pth")
print("Mirrored -> Models/brain_tumor_cnn_roi.pth")

# ── Test ──────────────────────────────────────────────────────────────────────
test_loss, test_acc = evaluate(test_loader)
print(f"\nTest accuracy : {test_acc:.2f}%  |  Test loss : {test_loss:.5f}")

# ── Curves ────────────────────────────────────────────────────────────────────
xs = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(xs, history["train_loss"], label="Train", color="#2196F3", lw=2)
ax1.plot(xs, history["val_loss"],   label="Val",   color="#F44336", lw=2)
ax1.axvline(stopper.best_epoch + 1, ls="--", color="#4CAF50", lw=1.5,
            label=f"Best (ep {stopper.best_epoch + 1})")
ax1.set_title("Improved CNN — Loss"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(xs, history["train_acc"], label="Train", color="#2196F3", lw=2)
ax2.plot(xs, history["val_acc"],   label="Val",   color="#F44336", lw=2)
ax2.set_title("Improved CNN — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Plots/improved_curves_roi.png", dpi=150); plt.close()
print("Saved: Plots/improved_curves_roi.png")
print(f"Best model: {CKPT}")
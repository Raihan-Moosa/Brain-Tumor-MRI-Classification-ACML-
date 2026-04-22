"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        BrainTumorNet-Lite — CPU-Optimised Brain MRI Classifier              ║
║   4 classes: glioma | meningioma | notumor | pituitary                     ║
║   Designed for: CPU-only training (no NVIDIA GPU required)                  ║
║   Target: 95%+ test accuracy | ~5–8 min/epoch on Intel i7                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Key optimisations for CPU training
  ✔ 128×128 input  →  4× less computation than 256×256
  ✔ Depthwise separable convolutions  →  ~8× faster than standard Conv2d
  ✔ ~2M parameters  →  vs 57M in the original version
  ✔ Retained: SE attention, residual connections, label smoothing,
              MixUp, cosine LR schedule, TTA, early stopping

Dependencies
  pip install torch torchvision matplotlib seaborn scikit-learn
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
#  0.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count())   # use all CPU cores

DEVICE = torch.device("cpu")           # Intel Iris Xe is not CUDA-capable
print(f"\n{'='*62}")
print(f"  Device  : CPU  ({os.cpu_count()} logical cores)")
print(f"  Mode    : BrainTumorNet-Lite  (CPU-optimised)")
print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  1.  PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Users\raiha\Downloads\MRI Tumor Dataset\split_70_15_15"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "best_braintumor_lite.pth")


# ──────────────────────────────────────────────────────────────────────────────
#  2.  HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = 128       # 128×128 — 4× less compute than 256×256
BATCH_SIZE      = 32        # larger batch = fewer steps per epoch = faster
NUM_CLASSES     = 4
NUM_EPOCHS      = 40
BASE_LR         = 3e-4
MIN_LR          = 1e-6
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA     = 0.30
MIN_DELTA       = 0.10      # minimum val-acc improvement to count as progress
PATIENCE        = 12        # early-stop after this many non-improving epochs
T0              = 10
T_MULT          = 2
NUM_WORKERS     = 0         # MUST be 0 on Windows (avoids multiprocess crash)
CLASS_NAMES     = ["glioma", "meningioma", "notumor", "pituitary"]

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
#  3.  DATA AUGMENTATION & LOADERS
# ──────────────────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0,
                             translate=(0.10, 0.10),
                             scale=(0.88, 1.12)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TTA_TRANSFORMS = [
    eval_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ]),
]

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)

print(f"Dataset  →  train: {len(train_ds):,}  |  val: {len(val_ds):,}  |  test: {len(test_ds):,}")
print(f"Classes  →  {train_ds.class_to_idx}")
print(f"Image size: {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH_SIZE}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  4.  MIXUP
# ──────────────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
#  5.  BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution = depthwise conv + pointwise conv.
    Approximates a standard Conv2d at ~8× lower compute cost.
    This is the core trick that makes the model CPU-feasible.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3,
                             stride=stride, padding=1,
                             groups=in_ch, bias=False)   # depthwise
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False) # pointwise
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class LiteResBlock(nn.Module):
    """
    Lightweight residual block using depthwise separable convolutions + SE.
    Much cheaper than the original ResBlock while keeping the residual structure.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)
        self.se    = SEBlock(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return F.relu(out + self.skip(x), inplace=True)


# ──────────────────────────────────────────────────────────────────────────────
#  6.  MODEL — BrainTumorNet-Lite
# ──────────────────────────────────────────────────────────────────────────────

class BrainTumorNetLite(nn.Module):
    """
    Lightweight custom CNN for 128×128 brain MRI classification.

    Spatial flow:
      128 → Stem → 32px → Stage1 → Stage2 → Stage3 → Stage4
          → 4px → GAP + GMP → FC head → 4 logits

    ~2M parameters — fast enough for CPU training.
    Keeps the key accuracy-boosting ideas: residual connections,
    SE attention, dual pooling fusion, strong augmentation.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # ── Stem: 128 → 32 ───────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),   # 64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # 32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # ── Stage 1: 64ch → 128ch, 32px ──────────────────────────────────────
        self.stage1 = nn.Sequential(
            LiteResBlock(64,  128, stride=1),
            LiteResBlock(128, 128, stride=1),
        )

        # ── Stage 2: 128ch → 256ch, 32 → 16px ───────────────────────────────
        self.stage2 = nn.Sequential(
            LiteResBlock(128, 256, stride=2),
            LiteResBlock(256, 256, stride=1),
            LiteResBlock(256, 256, stride=1),
        )

        # ── Stage 3: 256ch → 512ch, 16 → 8px ────────────────────────────────
        self.stage3 = nn.Sequential(
            LiteResBlock(256, 512, stride=2),
            LiteResBlock(512, 512, stride=1),
            LiteResBlock(512, 512, stride=1),
        )

        # ── Stage 4: 512ch → 512ch, 8 → 4px ─────────────────────────────────
        self.stage4 = nn.Sequential(
            LiteResBlock(512, 512, stride=2),
            LiteResBlock(512, 512, stride=1),
        )

        # ── Pooling fusion ────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)   # 512-d
        self.gmp = nn.AdaptiveMaxPool2d(1)   # 512-d  →  cat → 1024-d

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.cat([self.gap(x), self.gmp(x)], dim=1)
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
#  7.  LOSS
# ──────────────────────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.10):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_prob, self.smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_prob).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
#  8.  TRAIN / EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(1)
        correct    += (lam * (preds == y_a).float()
                       + (1 - lam) * (preds == y_b).float()).sum().item()
        total      += imgs.size(0)

        # Progress indicator every 50 batches so you know it's still running
        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={100*correct/total:.1f}%",
                  flush=True)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        logits      = model(imgs)
        total_loss += ce(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
#  9.  TEST-TIME AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model, dataset_root):
    model.eval()
    all_probs, all_labels = [], None
    for i, tf in enumerate(TTA_TRANSFORMS):
        ds     = datasets.ImageFolder(dataset_root, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
        probs_run, labels_run = [], []
        for imgs, labels in loader:
            probs_run.append(F.softmax(model(imgs), dim=1))
            if i == 0:
                labels_run.append(labels)
        all_probs.append(torch.cat(probs_run))
        if i == 0:
            all_labels = torch.cat(labels_run)
    avg_probs = torch.stack(all_probs).mean(0)
    return avg_probs.argmax(1).numpy(), all_labels.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# 10.  PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def save_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(history["val_loss"],   label="Val",   color="#F44336", lw=2)
    ax1.set_title("Loss per Epoch", fontsize=14)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", color="#2196F3", lw=2)
    ax2.plot(history["val_acc"],   label="Val",   color="#F44336", lw=2)
    ax2.axhline(95, color="#4CAF50", ls="--", lw=1.4, label="95% target")
    ax2.set_title("Accuracy per Epoch", fontsize=14)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved : {path}")


def save_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, norm], ["d", ".2f"],
        ["Confusion Matrix — Counts", "Confusion Matrix — Normalised"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, linecolor="gray")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title(title,        fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved : {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    model    = BrainTumorNetLite(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BrainTumorNet-Lite  →  {n_params:,} trainable parameters\n")

    criterion = LabelSmoothCE(smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(),
                            lr=BASE_LR,
                            weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=T0,
                                            T_mult=T_MULT,
                                            eta_min=MIN_LR)

    history    = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_val   = 0.0
    no_improve = 0

    hdr = (f"{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
           f"{'Va Loss':>8}  {'Va Acc':>7}  {'LR':>9}  {'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\n  [ Epoch {epoch}/{NUM_EPOCHS} ]")

        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader)
        scheduler.step(epoch)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr  = optimizer.param_groups[0]["lr"]
        tag = ""

        # Meaningful-delta early stopping
        if vl_acc > best_val + MIN_DELTA:
            best_val   = vl_acc
            no_improve = 0
            torch.save(model.state_dict(), CKPT_PATH)
            tag = "  ★ saved"
        else:
            no_improve += 1

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>6.2f}%  "
              f"{vl_loss:>8.4f}  {vl_acc:>6.2f}%  {lr:>9.2e}"
              f"  {elapsed:>5.0f}s{tag}")

        if no_improve >= PATIENCE:
            print(f"\n  ⚑ Early stop — no meaningful improvement "
                  f"for {PATIENCE} epochs.\n")
            break

    print(f"\n  Best val accuracy : {best_val:.2f}%")
    save_training_curves(history)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL TEST EVALUATION")
    print("=" * 62)

    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    _, std_acc = evaluate(model, test_loader)
    print(f"\n  Standard  test accuracy : {std_acc:.2f}%")

    y_pred, y_true = predict_with_tta(model, TEST_DIR)
    tta_acc = 100.0 * (y_pred == y_true).mean()
    print(f"  TTA       test accuracy : {tta_acc:.2f}%\n")

    print(classification_report(y_true, y_pred,
                                 target_names=CLASS_NAMES, digits=4))
    save_confusion_matrix(y_true, y_pred)

    print(f"\n  All outputs saved to: {OUT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
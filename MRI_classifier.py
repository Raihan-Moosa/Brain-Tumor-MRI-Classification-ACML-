#AttentionCNN
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

from stopping import EarlyStopping


#==Reproducibility================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count())
DEVICE = torch.device("cpu")

print(f"\n{'='*62}")
print(f"  Device  : CPU  ({os.cpu_count()} logical cores)")
print(f"  Model   : BrainTumorNet-Lite  (FIXED MIN_DELTA + EarlyStopping)")
print(f"{'='*62}\n")


#==Paths=======================================================================
BASE  = r"Dataset"
TRAIN = os.path.join(BASE, "train")
VAL   = os.path.join(BASE, "val")
TEST  = os.path.join(BASE, "test")

os.makedirs("Models",                   exist_ok=True)
os.makedirs("Models/saliency_ckpts",    exist_ok=True)   # sequential saliency
os.makedirs("Plots",                    exist_ok=True)

CKPT_PATH = "Models/best_braintumor_lite.pth"


#==Hyper-parameters============================================================
IMG_SIZE          = 128
BATCH_SIZE        = 32
NUM_CLASSES       = 4
NUM_EPOCHS        = 60

BASE_LR           = 3e-4
MIN_LR            = 1e-6
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.10
MIXUP_ALPHA       = 0.30

#==Early stopping=============================================================
ES_PATIENCE       = 5           # epochs of non-improvement before stopping
ES_DELTA          = 0.005

#==Sequential saliency========================================================
SALIENCY_INTERVAL = 5           # save a full checkpoint every N epochs

T0, T_MULT        = 10, 2
NUM_WORKERS       = 0
CLASS_NAMES       = ["glioma", "meningioma", "notumor", "pituitary"]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


#==Transforms=================================================================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.88, 1.12)),
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
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE+16, IMG_SIZE+16)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomRotation(degrees=(10, 10)),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]


#==Data loaders================================================================
train_ds = datasets.ImageFolder(TRAIN, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL,   transform=eval_tf)
test_ds  = datasets.ImageFolder(TEST,  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)

print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
print(f"Classes: {train_ds.class_to_idx}\n")


#==MixUp======================================================================
def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==Architecture — BrainTumorNet-Lite=========================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class LiteResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch,  out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)
        self.se    = SEBlock(out_ch)
        self.skip  = nn.Sequential()
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


class BrainTumorNetLite(nn.Module):
    """
    Grad-CAM target layer: model.stage4[1].conv2.pw
    (last pointwise conv before global pooling — highest-level spatial features)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(LiteResBlock(64,  128, stride=1),
                                    LiteResBlock(128, 128, stride=1))
        self.stage2 = nn.Sequential(LiteResBlock(128, 256, stride=2),
                                    LiteResBlock(256, 256, stride=1),
                                    LiteResBlock(256, 256, stride=1))
        self.stage3 = nn.Sequential(LiteResBlock(256, 512, stride=2),
                                    LiteResBlock(512, 512, stride=1),
                                    LiteResBlock(512, 512, stride=1))
        self.stage4 = nn.Sequential(LiteResBlock(512, 512, stride=2),
                                    LiteResBlock(512, 512, stride=1))
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.gmp  = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256, bias=False), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.40),
            nn.Linear(256,  128, bias=False), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.cat([self.gap(x), self.gmp(x)], dim=1)
        return self.head(x)


#==Loss========================================================================
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


#==Train / eval helpers========================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
        optimizer.zero_grad()
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
            if i == 0: labels_run.append(labels)
        all_probs.append(torch.cat(probs_run))
        if i == 0: all_labels = torch.cat(labels_run)
    avg = torch.stack(all_probs).mean(0)
    return avg.argmax(1).numpy(), all_labels.numpy()


#==Main training loop============================================================
def main():
    model     = BrainTumorNetLite(num_classes=NUM_CLASSES)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BrainTumorNet-Lite — {n_params:,} trainable parameters\n")

    criterion = LabelSmoothCE(smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0,
                                            T_mult=T_MULT, eta_min=MIN_LR)

    #==EarlyStopping=============================================================
    stopper = EarlyStopping(
        patience=ES_PATIENCE,
        delta=ES_DELTA,          
        checkpoint_path=CKPT_PATH,
        verbose=True,
        mode="min",
    )

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    hdr = (f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  "
           f"{'Va Loss':>9}  {'Va Acc':>8}  {'LR':>10}  {'ES':>8}  {'Time':>6}")
    print(hdr); print("─" * len(hdr))

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader)
        scheduler.step(epoch)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        marker  = f"  {stopper.counter}/{ES_PATIENCE}"

        print(f"{epoch:>6}  {tr_loss:>9.5f}  {tr_acc:>7.2f}%  "
              f"{vl_loss:>9.5f}  {vl_acc:>7.2f}%  {lr:>10.2e}"
              f"  {marker}  {elapsed:>5.0f}s")

        #====Sequential saliency checkpoint======================================
        # Save a snapshot every SALIENCY_INTERVAL epochs regardless of whether
        # it is the best model. gradcam_sequential.py loads these in order to
        # visualise how the model's attention evolves during training.
        if epoch % SALIENCY_INTERVAL == 0:
            ckpt_name = f"Models/saliency_ckpts/epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_name)
            print(f"  [Saliency ckpt] Saved: {ckpt_name}")

        #====Adaptive early stopping=============================================
        if stopper.step(vl_loss, model):
            print(f"\n  ⚑  Early stop at epoch {epoch}. "
                  f"Best val loss {stopper.best_score:.5f} "
                  f"(epoch {stopper.best_epoch + 1}).\n")
            # Save one final saliency checkpoint at the stopping epoch
            torch.save(model.state_dict(),
                       f"Models/saliency_ckpts/epoch_{epoch:03d}_final.pth")
            break

    # Always restore best checkpoint — not the final overfitted epoch
    stopper.load_best(model)
    print(f"\n  Best val accuracy — check curves above.")

    
    #==Plots====================================================================
    epochs_ran = len(history["train_loss"])
    xs = range(1, epochs_ran + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(xs, history["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(xs, history["val_loss"],   label="Val",   color="#F44336", lw=2)
    if stopper.best_epoch < epochs_ran:
        ax1.axvline(stopper.best_epoch + 1, ls="--", color="#4CAF50", lw=1.5,
                    label=f"Best (ep {stopper.best_epoch+1})")
    ax1.set_title("BrainTumorNet-Lite — Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(xs, history["train_acc"], label="Train", color="#2196F3", lw=2)
    ax2.plot(xs, history["val_acc"],   label="Val",   color="#F44336", lw=2)
    ax2.axhline(95, color="#FF9800", ls="--", lw=1.4, label="95% target")
    ax2.set_title("BrainTumorNet-Lite — Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Plots/lite_curves.png", dpi=150); plt.close()
    print("Saved: Plots/lite_curves.png")


    #==Test evaluation==========================================================
    print("\n" + "=" * 62)
    print("  FINAL TEST EVALUATION")
    print("=" * 62)

    _, std_acc = evaluate(model, test_loader)
    print(f"\n  Standard test accuracy : {std_acc:.2f}%")

    y_pred, y_true = predict_with_tta(model, TEST)
    tta_acc = 100.0 * (y_pred == y_true).mean()
    print(f"  TTA      test accuracy : {tta_acc:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    #=Confusion matrix==========================================================
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, norm], ["d", ".2f"],
        ["Confusion Matrix — Counts", "Confusion Matrix — Normalised"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig("Plots/lite_confusion.png", dpi=150); plt.close()
    print("Saved: Plots/lite_confusion.png")
    print(f"\nBest model: {CKPT_PATH}")
    print(f"Saliency checkpoints: Models/saliency_ckpts/\n")


if __name__ == "__main__":
    main()

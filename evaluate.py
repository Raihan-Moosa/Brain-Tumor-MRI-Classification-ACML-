"""
evaluate_all.py  —  Unified evaluation for all models and datasets
==================================================================
Evaluates all 6 combinations (3 models x raw + roi) and writes
a results file and confusion matrix plot for each.

Usage
-----
  python evaluate_all.py                         # all 6 combinations
  python evaluate_all.py --model baseline       # one model, both datasets
  python evaluate_all.py --model improved --dataset roi
  python evaluate_all.py --summary               # print comparison table only

Outputs
-------
  Results/<model>_<dataset>.txt     classification report + accuracy
  Plots/<model>_<dataset>_confusion.png
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("Results", exist_ok=True)
os.makedirs("Plots",   exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DATASET_ROOTS = {
    "raw": "Dataset/test",
    "roi": "Dataset/ROI/test",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Architectures — must match the corresponding training scripts exactly
# ─────────────────────────────────────────────────────────────────────────────

class Baseline(nn.Module):
    """Baseline (Baseline) — no BatchNorm, AdaptiveAvgPool(4,4)."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,  32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


class Improved(nn.Module):
    """Improved (Improved) — BatchNorm, global pooling, wider head."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  3, padding=1), nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


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
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)

class LiteResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch,  out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.se    = SEBlock(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        return F.relu(self.se(self.conv2(self.conv1(x))) + self.skip(x),
                      inplace=True)

class Lite(nn.Module):
    """Lite (Lite) — depthwise sep convs, SE attention, residuals."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(LiteResBlock(64,  128),
                                    LiteResBlock(128, 128))
        self.stage2 = nn.Sequential(LiteResBlock(128, 256, stride=2),
                                    LiteResBlock(256, 256),
                                    LiteResBlock(256, 256))
        self.stage3 = nn.Sequential(LiteResBlock(256, 512, stride=2),
                                    LiteResBlock(512, 512),
                                    LiteResBlock(512, 512))
        self.stage4 = nn.Sequential(LiteResBlock(512, 512, stride=2),
                                    LiteResBlock(512, 512))
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
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.head(torch.cat([self.gap(x), self.gmp(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "baseline": {
        "cls":      Baseline,
        "weights":  {"raw": "Models/best_baseline.pth",
                     "roi": "Models/best_baseline_roi.pth"},
        "sizes":    {"raw": 512, "roi": 224},
        "mean":     None,   # no normalisation
        "display":  "Baseline (Baseline)",
    },
    "improved": {
        "cls":      Improved,
        "weights":  {"raw": "Models/best_improved.pth",
                     "roi": "Models/best_improved_roi.pth"},
        "sizes":    {"raw": 256, "roi": 224},
        "mean":     [0.5, 0.5, 0.5],
        "std":      [0.5, 0.5, 0.5],
        "display":  "Improved (Improved)",
    },
    "lite": {
        "cls":      Lite,
        "weights":  {"raw": "Models/best_braintumor_lite.pth",
                     "roi": "Models/best_braintumor_lite_roi.pth"},
        "sizes":    {"raw": 128, "roi": 224},
        "mean":     [0.485, 0.456, 0.406],
        "std":      [0.229, 0.224, 0.225],
        "display":  "Lite (Lite)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(cfg: dict, dataset_key: str) -> transforms.Compose:
    tf = [
        transforms.Resize((cfg["sizes"][dataset_key], cfg["sizes"][dataset_key])),
        transforms.ToTensor(),
    ]
    if cfg.get("mean"):
        tf.append(transforms.Normalize(cfg["mean"], cfg["std"]))
    return transforms.Compose(tf)


@torch.no_grad()
def run_evaluation(model_key: str, dataset_key: str) -> dict | None:
    cfg      = MODEL_REGISTRY[model_key]
    wpath    = cfg["weights"][dataset_key]

    if not os.path.exists(wpath):
        print(f"  SKIPPED — weights not found: {wpath}")
        return None

    if not os.path.exists(DATASET_ROOTS[dataset_key]):
        print(f"  SKIPPED — dataset not found: {DATASET_ROOTS[dataset_key]}")
        return None

    model = cfg["cls"]().to(DEVICE)
    model.load_state_dict(torch.load(wpath, map_location=DEVICE,
                                      weights_only=True))
    model.eval()

    tf      = get_transform(cfg, dataset_key)
    ds      = datasets.ImageFolder(DATASET_ROOTS[dataset_key], transform=tf)
    loader  = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                    target_names=CLASS_NAMES, digits=4)
    cm     = confusion_matrix(all_labels, all_preds)

    return dict(acc=acc, report=report, cm=cm,
                labels=all_labels, preds=all_preds)


def save_results(model_key: str, dataset_key: str, results: dict):
    cfg   = MODEL_REGISTRY[model_key]
    label = f"{model_key}_{dataset_key}"

    # Text report
    txt_path = f"Results/{label}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Model  : {cfg['display']}\n")
        f.write(f"Dataset: {dataset_key.upper()}\n")
        f.write(f"Weights: {cfg['weights'][dataset_key]}\n")
        f.write(f"Test Accuracy: {results['acc']:.4f}  "
                f"({results['acc']*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(results["report"])
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(results["cm"]))

    # Confusion matrix plot
    cm      = results["cm"]
    norm_cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, norm_cm],
        ["d", ".2f"],
        ["Counts", "Normalised"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=data.max())
        ax.set_title(f"Confusion Matrix — {title}", fontsize=12)
        plt.colorbar(im, ax=ax)
        ticks = range(len(CLASS_NAMES))
        ax.set_xticks(ticks); ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_yticks(ticks); ax.set_yticklabels(CLASS_NAMES)
        thresh = data.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.suptitle(f"{cfg['display']}  —  {dataset_key.upper()} dataset  "
                 f"({results['acc']*100:.2f}% accuracy)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"Plots/{label}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Accuracy : {results['acc']*100:.2f}%")
    print(f"  Saved    : Results/{label}.txt")
    print(f"  Saved    : Plots/{label}_confusion.png")
    return results["acc"]


# ─────────────────────────────────────────────────────────────────────────────
#  Summary comparison table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(scores: dict):
    print("\n" + "=" * 58)
    print("  RESULTS SUMMARY")
    print("=" * 58)
    print(f"  {'Model':<22}  {'Raw':>10}  {'ROI':>10}  {'Δ':>8}")
    print("  " + "-" * 54)
    for mk, cfg in MODEL_REGISTRY.items():
        raw = scores.get((mk, "raw"))
        roi = scores.get((mk, "roi"))
        raw_str = f"{raw*100:.2f}%" if raw is not None else "—"
        roi_str = f"{roi*100:.2f}%" if roi is not None else "—"
        if raw is not None and roi is not None:
            delta = (roi - raw) * 100
            delta_str = f"{delta:+.2f}%"
        else:
            delta_str = "—"
        print(f"  {cfg['display']:<22}  {raw_str:>10}  {roi_str:>10}  {delta_str:>8}")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all models on raw and/or ROI datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   default="all",
                        choices=["baseline", "improved", "lite", "all"])
    parser.add_argument("--dataset", default="both",
                        choices=["raw", "roi", "both"])
    parser.add_argument("--summary", action="store_true",
                        help="Print comparison table from saved Results/ files.")
    args = parser.parse_args()

    models   = list(MODEL_REGISTRY.keys()) if args.model == "all" \
               else [args.model]
    datasets_ = ["raw", "roi"] if args.dataset == "both" \
                else [args.dataset]

    scores = {}

    for mk in models:
        for dk in datasets_:
            cfg = MODEL_REGISTRY[mk]
            print(f"\n[{cfg['display']} / {dk.upper()}]")
            results = run_evaluation(mk, dk)
            if results:
                acc = save_results(mk, dk, results)
                scores[(mk, dk)] = acc

    if scores:
        print_summary(scores)

    print("Done.")
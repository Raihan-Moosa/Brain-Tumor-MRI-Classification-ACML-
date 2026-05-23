"""
gradcam_all.py  —  Unified Grad-CAM for all models and datasets
================================================================
Replaces gradcam_3d.py, gradcam_visual.py, and gradcam_lite.py.
One script handles all 6 combinations:

  Model:    SimpleCNN (baseline) | NormCNN (improved) | AttentionCNN (lite)
  Dataset:  raw | roi

Usage
-----
  # All models, raw dataset
  python gradcam_all.py

  # Specific model and dataset
  python gradcam_all.py --model normcnn --dataset roi

  # Sequential saliency filmstrip (AttentionCNN only, needs saliency_ckpts/)
  python gradcam_all.py --model attentioncnn --sequential

  # Run everything in one go
  python gradcam_all.py --all

Arguments
---------
  --model       simplecnn | normcnn | attentioncnn | all  (default: all)
  --dataset     raw | roi | both                          (default: raw)
  --sequential  filmstrip across training checkpoints (attentioncnn only)
  --all         equivalent to --model all --dataset both

Outputs
-------
  Plots/gradcam/<model>_<dataset>/gradcam_<class>.png
  Plots/gradcam/<model>_<dataset>/gradcam_summary.png
  Plots/gradcam/attentioncnn_<dataset>/sequential_<class>.png

Architecture registry
---------------------
Each model is registered with its weight path, target layer accessor,
image size, and normalisation so the hook logic is identical for all.
Adding a new model requires only a new entry in MODEL_REGISTRY.
"""

import argparse
import os
import re
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOTS = {
    "raw": "Dataset/test",
    "roi": "Dataset/ROI/test",
}

SALIENCY_DIR   = "Models/saliency_ckpts"
OUT_ROOT       = "Plots/gradcam"
CLASS_NAMES    = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE         = torch.device("cpu")
BG             = "#111111"
ATTENTION_PCT  = 70     # top 30% of activations shown in threshold panel

# Clinical assessment — update after ROI retraining if findings change
CLINICAL_NOTES = {
    "raw": {
        "glioma":     "⚠ Fires on image border artifact — NOT tumour",
        "meningioma": "✓ Activates skull base — anatomically correct",
        "notumor":    "✓ Diffuse central activation — correct",
        "pituitary":  "✓ Activates brain base — anatomically correct",
    },
    "roi": {
        "glioma":     "Post-ROI: border removed — check if attention shifted",
        "meningioma": "✓ Post-ROI activation — verify anatomical region",
        "notumor":    "✓ Post-ROI activation — verify central focus",
        "pituitary":  "✓ Post-ROI activation — verify brain base focus",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Model architectures
# ─────────────────────────────────────────────────────────────────────────────

# ── SimpleCNN (Baseline) ──────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    4-block plain CNN. No BatchNorm, no normalisation, no attention.
    Grad-CAM target: features[9]  (last Conv2d, 64->128)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [0-2]
            nn.Conv2d(16,  32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [3-5]
            nn.Conv2d(32,  64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [6-8]
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [9-11]
            nn.AdaptiveAvgPool2d((4, 4)),                                     # [12]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── NormCNN (Improved) ───────────────────────────────────────────────────────
class NormCNN(nn.Module):
    """
    4-block CNN with BatchNorm and normalised inputs.
    Grad-CAM target: features[12]  (last Conv2d, 64->128; BN shifts index)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   16,  3, padding=1), nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2),  # [0-3]
            nn.Conv2d(16,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),  # [4-7]
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # [8-11]
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # [12-15]
            nn.AdaptiveAvgPool2d((1, 1)),                                                          # [16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── AttentionCNN (Lite) ───────────────────────────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
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
        return F.relu(self.se(self.conv2(self.conv1(x))) + self.skip(x), inplace=True)

class AttentionCNN(nn.Module):
    """
    Depthwise-separable residual CNN with SE channel attention.
    Grad-CAM target: stage4[1].conv2.pw  (last pointwise conv before pooling)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(LiteResBlock(64,  128), LiteResBlock(128, 128))
        self.stage2 = nn.Sequential(LiteResBlock(128, 256, stride=2), LiteResBlock(256, 256), LiteResBlock(256, 256))
        self.stage3 = nn.Sequential(LiteResBlock(256, 512, stride=2), LiteResBlock(512, 512), LiteResBlock(512, 512))
        self.stage4 = nn.Sequential(LiteResBlock(512, 512, stride=2), LiteResBlock(512, 512))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.gmp    = nn.AdaptiveMaxPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.40),
            nn.Linear(256,  128, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.head(torch.cat([self.gap(x), self.gmp(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
#  Each entry defines everything needed to load and hook the model.
#  To add a new model: add one entry here, nothing else changes.
# ─────────────────────────────────────────────────────────────────────────────
def _get_simplecnn_target(m):  return m.features[9]
def _get_normcnn_target(m):    return m.features[12]
def _get_attentioncnn_target(m): return m.stage4[1].conv2.pw

MODEL_REGISTRY = {
    "simplecnn": {
        "cls":         SimpleCNN,
        "weights":     {
            "raw": "Models/best_baseline.pth",
            "roi": "Models/best_baseline_roi.pth",
        },
        "target_fn":   _get_simplecnn_target,
        "img_size":    512,
        "mean":        [0.0, 0.0, 0.0],   # no normalisation in baseline
        "std":         [1.0, 1.0, 1.0],
        "display":     "SimpleCNN (Baseline)",
    },
    "normcnn": {
        "cls":         NormCNN,
        "weights":     {
            "raw": "Models/best_improved.pth",
            "roi": "Models/best_improved_roi.pth",
        },
        "target_fn":   _get_normcnn_target,
        "img_size":    256,
        "mean":        [0.5, 0.5, 0.5],
        "std":         [0.5, 0.5, 0.5],
        "display":     "NormCNN (Improved)",
    },
    "attentioncnn": {
        "cls":         AttentionCNN,
        "weights":     {
            "raw": "Models/best_braintumor_lite.pth",
            "roi": "Models/best_braintumor_lite_roi.pth",
        },
        "target_fn":   _get_attentioncnn_target,
        "img_size":    128,
        "mean":        [0.485, 0.456, 0.406],
        "std":         [0.229, 0.224, 0.225],
        "display":     "AttentionCNN (Lite)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  From-scratch Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    From-scratch Grad-CAM via register_forward_hook + register_full_backward_hook.
    No Captum. Works on any target layer passed at construction time.

    Mathematics:
      alpha_k = (1/Z) * sum_ij ( d y^c / d A^k_ij )   [GAP of gradients]
      L = ReLU( sum_k alpha_k * A^k )
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handles    = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            ),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            ),
        ]

    def generate(self, inp: torch.Tensor,
                 class_idx: int = None) -> tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        inp    = inp.clone().requires_grad_(True)
        logits = self.model(inp)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        if class_idx is None:
            class_idx = int(probs.argmax())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        alpha = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam   = F.relu((alpha * self.activations).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, size=(inp.shape[2], inp.shape[3]),
                              mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        cam   = (cam - lo) / (hi - lo + 1e-8)
        return cam, class_idx, probs

    def remove(self):
        for h in self._handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_key: str, dataset_key: str) -> nn.Module:
    cfg    = MODEL_REGISTRY[model_key]
    wpath  = cfg["weights"][dataset_key]
    if not Path(wpath).exists():
        raise FileNotFoundError(
            f"Weights not found: {wpath}\n"
            f"Train {model_key} on {dataset_key} dataset first."
        )
    model = cfg["cls"]().to(DEVICE)
    model.load_state_dict(torch.load(wpath, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def get_transform(model_key: str) -> transforms.Compose:
    cfg = MODEL_REGISTRY[model_key]
    tf  = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ]
    if cfg["mean"] != [0.0, 0.0, 0.0]:
        tf.append(transforms.Normalize(cfg["mean"], cfg["std"]))
    return transforms.Compose(tf)


def denorm(tensor: torch.Tensor, model_key: str) -> np.ndarray:
    cfg = MODEL_REGISTRY[model_key]
    t   = tensor.squeeze().cpu().clone()
    for i, (m, s) in enumerate(zip(cfg["mean"], cfg["std"])):
        t[i] = t[i] * s + m
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def pick_one_per_class(model: nn.Module, model_key: str,
                       dataset_key: str, seed: int = 42) -> dict:
    random.seed(seed)
    tf = get_transform(model_key)
    ds = datasets.ImageFolder(DATASET_ROOTS[dataset_key], transform=tf)

    by_class = {c: [] for c in range(4)}
    for idx, (_, lbl) in enumerate(ds.samples):
        by_class[lbl].append(idx)
    for c in range(4):
        random.shuffle(by_class[c])

    picked = {}
    for cls in range(4):
        for idx in by_class[cls]:
            img_t, true_lbl = ds[idx]
            with torch.no_grad():
                probs = F.softmax(model(img_t.unsqueeze(0)), dim=1)
            pred = probs.argmax(1).item()
            if pred == true_lbl:
                picked[cls] = dict(img_t=img_t, true_label=true_lbl,
                                   pred_label=pred,
                                   confidence=probs[0, pred].item())
                break
        if cls not in picked:
            idx   = by_class[cls][0]
            img_t, true_lbl = ds[idx]
            with torch.no_grad():
                probs = F.softmax(model(img_t.unsqueeze(0)), dim=1)
            pred = probs.argmax(1).item()
            picked[cls] = dict(img_t=img_t, true_label=true_lbl,
                               pred_label=pred,
                               confidence=probs[0, pred].item())
    return picked


# ─────────────────────────────────────────────────────────────────────────────
#  Panel builders (interpretable layout — no 3D surface)
# ─────────────────────────────────────────────────────────────────────────────
def _panel_original(ax, img_np, true_label):
    ax.imshow(img_np)
    ax.set_title("Input MRI", color="white", fontsize=10, pad=5)
    ax.set_xlabel(f"True: {CLASS_NAMES[true_label].upper()}",
                  color="#aaaaaa", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _panel_overlay(ax, img_np, cam):
    ax.imshow(img_np)
    mp = ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title("Grad-CAM\n(where the model looked)",
                 color="white", fontsize=10, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(mp, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Activation strength", color="#aaaaaa", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#aaaaaa", labelcolor="#aaaaaa", labelsize=6)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0\n(ignored)", "0.5", "1\n(fired hard)"])


def _panel_threshold(ax, img_np, cam, true_label, pred_label, dataset_key):
    mask = (cam >= np.percentile(cam, ATTENTION_PCT)).astype(np.uint8)
    ax.imshow(img_np)
    overlay = np.zeros((*cam.shape, 4), dtype=np.float32)
    overlay[mask == 1] = [1.0, 1.0, 1.0, 0.35]
    ax.imshow(overlay)
    ax.contour(mask, levels=[0.5], colors=["#FFD700"], linewidths=[1.5])
    note   = CLINICAL_NOTES[dataset_key].get(CLASS_NAMES[true_label], "")
    colour = "#4ade80" if "✓" in note else "#f87171" if "⚠" in note else "#aaaaaa"
    ax.set_title("Attention Region\n(top 30% activations)",
                 color="white", fontsize=10, pad=5)
    ax.set_xlabel(note, color=colour, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def _panel_confidence(ax, probs, true_label, pred_label):
    colours = []
    for i in range(4):
        if i == true_label == pred_label: colours.append("#4ade80")
        elif i == true_label:             colours.append("#60a5fa")
        elif i == pred_label:             colours.append("#f87171")
        else:                             colours.append("#444444")

    bars = ax.barh(range(4), probs * 100, color=colours,
                   height=0.55, edgecolor="none")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="white", fontsize=8)

    ax.set_yticks(range(4))
    ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES],
                       color="white", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#aaaaaa", fontsize=8)
    ax.set_title("Prediction Confidence\n(all 4 classes)",
                 color="white", fontsize=10, pad=5)
    ax.tick_params(axis="x", colors="#aaaaaa", labelsize=7)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_edgecolor("#444444")
    ax.set_facecolor(BG)
    ax.legend(handles=[
        mpatches.Patch(color="#4ade80", label="Correct"),
        mpatches.Patch(color="#f87171", label="Wrong prediction"),
        mpatches.Patch(color="#60a5fa", label="True (not predicted)"),
    ], fontsize=6, framealpha=0.2, labelcolor="white")


def make_figure(img_t, cam, probs, true_label, pred_label,
                model_key, dataset_key, tag="") -> plt.Figure:
    img_np = denorm(img_t, model_key)
    fig    = plt.figure(figsize=(18, 5), facecolor=BG)
    gs     = gridspec.GridSpec(1, 4, figure=fig, wspace=0.28,
                               left=0.03, right=0.97, top=0.80, bottom=0.12)
    for i, bg in enumerate([BG, BG, BG, None]):
        ax = fig.add_subplot(gs[i])
        if bg: ax.set_facecolor(bg)

    axes = [fig.axes[-4], fig.axes[-3], fig.axes[-2], fig.axes[-1]]
    _panel_original(axes[0], img_np, true_label)
    _panel_overlay(axes[1], img_np, cam)
    _panel_threshold(axes[2], img_np, cam, true_label, pred_label, dataset_key)
    _panel_confidence(axes[3], probs, true_label, pred_label)

    correct = "✓ Correct" if true_label == pred_label else "✗ Misclassification"
    col     = "#4ade80" if true_label == pred_label else "#f87171"
    cfg     = MODEL_REGISTRY[model_key]
    title   = (f"{CLASS_NAMES[true_label].upper()}  |  "
               f"Pred: {CLASS_NAMES[pred_label].upper()}  —  {correct}  "
               f"[{cfg['display']} / {dataset_key.upper()}]"
               + (f"  {tag}" if tag else ""))
    fig.suptitle(title, color=col, fontsize=12, fontweight="bold", y=0.96)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 1 — Standard Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def run_standard(model_key: str, dataset_key: str):
    cfg     = MODEL_REGISTRY[model_key]
    out_dir = Path(OUT_ROOT) / f"{model_key}_{dataset_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{cfg['display']} / {dataset_key.upper()}]")

    try:
        model = load_model(model_key, dataset_key)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return

    gradcam = GradCAM(model, cfg["target_fn"](model))
    picked  = pick_one_per_class(model, model_key, dataset_key)

    for cls in range(4):
        info  = picked[cls]
        cam, pred_idx, probs = gradcam.generate(
            info["img_t"].unsqueeze(0), class_idx=info["pred_label"]
        )
        fig  = make_figure(info["img_t"], cam, probs,
                           info["true_label"], pred_idx,
                           model_key, dataset_key)
        path = out_dir / f"gradcam_{CLASS_NAMES[cls]}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved: {path}")

    # Summary — 2-row grid (overlay + threshold mask)
    fig_s, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG)
    for col, cls in enumerate(range(4)):
        info  = picked[cls]
        cam, pred_idx, probs = gradcam.generate(
            info["img_t"].unsqueeze(0), class_idx=info["pred_label"]
        )
        img_np = denorm(info["img_t"], model_key)
        H, W   = cam.shape
        axes[0, col].set_facecolor(BG); axes[1, col].set_facecolor(BG)
        axes[0, col].imshow(img_np)
        axes[0, col].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1,
                             extent=[0, W, H, 0])
        axes[0, col].set_title(CLASS_NAMES[cls].upper(),
                               color="white", fontsize=12, fontweight="bold")
        axes[0, col].axis("off")
        mask = (cam >= np.percentile(cam, ATTENTION_PCT)).astype(np.uint8)
        axes[1, col].imshow(img_np)
        ov = np.zeros((*cam.shape, 4), dtype=np.float32)
        ov[mask == 1] = [1, 1, 1, 0.35]
        axes[1, col].imshow(ov)
        axes[1, col].contour(mask, levels=[0.5], colors=["#FFD700"], linewidths=[1.5])
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Grad-CAM Overlay",       color="white", fontsize=10)
    axes[1, 0].set_ylabel("Attention Region (top 30%)", color="white", fontsize=10)
    fig_s.suptitle(f"Grad-CAM Summary — {cfg['display']} / {dataset_key.upper()}",
                   color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "gradcam_summary.png"
    fig_s.savefig(path, dpi=150, bbox_inches="tight",
                  facecolor=fig_s.get_facecolor())
    plt.close(fig_s)
    print(f"    Saved: {path}")
    gradcam.remove()


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 2 — Sequential saliency (AttentionCNN only)
# ─────────────────────────────────────────────────────────────────────────────
def run_sequential(dataset_key: str):
    model_key = "attentioncnn"
    cfg       = MODEL_REGISTRY[model_key]
    out_dir   = Path(OUT_ROOT) / f"{model_key}_{dataset_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        Path(SALIENCY_DIR).glob("epoch_*.pth"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)),
    )
    if not ckpts:
        print(f"No checkpoints in {SALIENCY_DIR}. Run MRI_classifier.py first.")
        return

    print(f"\n  [Sequential saliency — AttentionCNN / {dataset_key.upper()}]")
    print(f"  Found {len(ckpts)} checkpoints.")

    # Use best-model's sample selection for consistent images across epochs
    ref_model = load_model(model_key, dataset_key)
    picked    = pick_one_per_class(ref_model, model_key, dataset_key)

    for cls in range(4):
        info   = picked[cls]
        img_t  = info["img_t"]
        img_np = denorm(img_t, model_key)
        n      = len(ckpts)

        fig = plt.figure(figsize=(15, 3.8 * n), facecolor=BG)
        gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.30,
                                left=0.07, right=0.97, top=0.96, bottom=0.02)

        for col, title in enumerate(["Grad-CAM Overlay",
                                      "Attention Region (top 30%)",
                                      "Prediction Confidence"]):
            ax = fig.add_subplot(gs[0, col])
            ax.set_title(title, color="white", fontsize=10, pad=5)
            ax.set_facecolor(BG)

        for row, ckpt in enumerate(ckpts):
            epoch_num = re.search(r"epoch_(\d+)", ckpt.name).group(1)
            m         = cfg["cls"]().to(DEVICE)
            m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE,
                                          weights_only=True))
            m.eval()
            gc  = GradCAM(m, cfg["target_fn"](m))
            cam, pred_idx, probs = gc.generate(img_t.unsqueeze(0), class_idx=cls)
            gc.remove()

            fig.text(0.01, 1 - (row + 0.5) / n, f"Epoch\n{epoch_num}",
                     color="white", fontsize=8, va="center",
                     transform=fig.transFigure)

            ax_ov = fig.add_subplot(gs[row, 0]); ax_ov.set_facecolor(BG)
            ax_th = fig.add_subplot(gs[row, 1]); ax_th.set_facecolor(BG)
            ax_cf = fig.add_subplot(gs[row, 2])

            _panel_overlay(ax_ov, img_np, cam)
            _panel_threshold(ax_th, img_np, cam, info["true_label"],
                             pred_idx, dataset_key)
            _panel_confidence(ax_cf, probs, info["true_label"], pred_idx)
            if row > 0:
                ax_ov.set_title(""); ax_th.set_title("")

            conf = probs[pred_idx] * 100
            print(f"    [{CLASS_NAMES[cls]}] ep {epoch_num:>3}  "
                  f"pred={CLASS_NAMES[pred_idx]:<12}  conf={conf:.1f}%")

        fig.suptitle(
            f"Sequential Saliency — {CLASS_NAMES[cls].upper()}  "
            f"[AttentionCNN / {dataset_key.upper()}]",
            color="white", fontsize=12, fontweight="bold",
        )
        path = out_dir / f"sequential_{CLASS_NAMES[cls]}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved: {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Grad-CAM for SimpleCNN / NormCNN / AttentionCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default="all",
                        choices=["simplecnn", "normcnn", "attentioncnn", "all"])
    parser.add_argument("--dataset",    default="raw",
                        choices=["raw", "roi", "both"])
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential saliency filmstrip (AttentionCNN only)")
    parser.add_argument("--all",        action="store_true",
                        help="Run all models on both datasets")
    args = parser.parse_args()

    models   = list(MODEL_REGISTRY.keys()) if (args.model == "all" or args.all) \
               else [args.model]
    datasets_ = ["raw", "roi"] if (args.dataset == "both" or args.all) \
                else [args.dataset]

    if args.sequential:
        for ds in datasets_:
            run_sequential(ds)
    else:
        for mk in models:
            for dk in datasets_:
                run_standard(mk, dk)

    print("\nDone.")
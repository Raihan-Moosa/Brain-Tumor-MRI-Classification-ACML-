# Brain Tumor MRI Classification — How to Run

## 1. Environment Setup

### Prerequisites
- Python 3.9 or 3.11 (tested with 3.11)
- Virtual environment (venv or conda)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # macOS/Linux

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy==1.26.4        # CRITICAL: numpy<2 required for torch compatibility
pip install scikit-learn matplotlib seaborn opencv-python
pip install scipy pillow
```

**Important:** NumPy must be version < 2 (1.26.4 recommended) due to PyTorch compatibility.

## 2. Dataset Structure

The dataset must be organized as follows:

```
Dataset/
├── train/
│   ├── glioma/          (training images for glioma tumors)
│   ├── meningioma/      (training images for meningioma tumors)
│   ├── notumor/         (training images with no tumor)
│   └── pituitary/       (training images for pituitary tumors)
├── val/                 (same 4 subdirectories as train)
└── test/                (same 4 subdirectories as train)
```

Each subdirectory should contain MRI scan images (jpg/png format).

## 3. Main Training Scripts

### Option A: Improved Model (Recommended)
```bash
python MRI_classifier.py
```
- **Model:** BrainTumorNet-Lite (custom CNN with SE blocks, MixUp augmentation, cosine annealing)
- **Outputs:**
  - `Models/best_braintumor_lite.pth` — Best model checkpoint
  - `Models/saliency_ckpts/` — Intermediate checkpoints (every 5 epochs)
  - `Plots/` — Training/validation curves and confusion matrices
  - `Results/test_results.txt` — Final test set metrics
- **Features:**
  - Early stopping (patience=5, delta=0.005 on validation loss)
  - Test-time augmentation (TTA)
  - Automatic model checkpointing
- **Runtime:** ~10-60 minutes on CPU (depends on dataset size)

### Option B: Baseline CNN
```bash
python train_baseline.py
```
- **Model:** Simple baseline CNN (4 conv blocks, fixed linear layer)
- **Outputs:**
  - `Models/best_baseline.pth` — Best model checkpoint
  - `Plots/` — Training curves
  - `Results/test_results.txt` — Test metrics
- **Runtime:** ~5-30 minutes on CPU

### Option C: Alternative Improved Model
```bash
python train_improved.py
```
- Similar architecture to Option A with different hyperparameters
- Outputs to same directories as Option A

## 4. Evaluation & Analysis Scripts

### Evaluate Trained Model
```bash
python evaluate.py
```
- Loads the best model and evaluates on test set
- Generates:
  - Classification report (precision, recall, F1)
  - Confusion matrix visualization
  - Test metrics saved to `Results/test_results.txt`

### Grad-CAM Visualization
```bash
python gradcam_visual.py
```
- Generates saliency maps showing model attention regions
- Outputs to `Plots/gradcam/`

### 3D Grad-CAM
```bash
python gradcam_3d.py
```
- Extended Grad-CAM analysis for 3D visualization

## 5. Dataset Preparation (If needed)

If you need to split raw data into train/val/test:

```bash
python split_dataset.py
```
- Modifies the `SPLIT_RATIOS` in the script (default: 70% train, 15% val, 15% test)

## 6. Expected Output Files

After running `MRI_classifier.py`, expect:

```
Models/
├── best_braintumor_lite.pth       (saved model)
└── saliency_ckpts/
    ├── ckpt_epoch_5.pth
    ├── ckpt_epoch_10.pth
    └── ...

Plots/
├── training_validation_curves.png
├── confusion_matrix.png
└── gradcam/
    └── (saliency visualizations)

Results/
└── test_results.txt               (accuracy, precision, recall, F1)
```

## 7. Key Parameters

Edit these values in `MRI_classifier.py` if needed:

- `IMG_SIZE = 128` — Input image size (resized from original)
- `BATCH_SIZE = 32` — Batch size for training
- `NUM_EPOCHS = 60` — Maximum training epochs (early stopping will fire before this)
- `BASE_LR = 3e-4` — Initial learning rate
- `ES_PATIENCE = 5` — Early stopping patience (epochs)
- `ES_DELTA = 0.005` — Minimum improvement threshold

## 8. Troubleshooting

**Q: "ModuleNotFoundError: No module named 'numpy.exceptions'"**
- A: NumPy version too old. Run: `pip install numpy==1.26.4`

**Q: "RuntimeError: Numpy is not available"**
- A: NumPy 2.x incompatible with current PyTorch. Run: `pip install "numpy<2"`

**Q: GPU not detected (CUDA)**
- A: Scripts default to CPU. To use GPU, modify `DEVICE` in main scripts or install CPU-only PyTorch.

**Q: Out of memory on CPU**
- A: Reduce `BATCH_SIZE` (e.g., 16 or 8) in the script.

**Q: Training is very slow**
- A: This is expected on CPU. Consider:
  - Reducing dataset size
  - Using GPU if available
  - Reducing `NUM_EPOCHS`

## 9. Quick Start (30 seconds)

```bash
# 1. Setup environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install torch torchvision numpy==1.26.4 scikit-learn matplotlib seaborn

# 3. Run training
python MRI_classifier.py

# 4. Check results
# → Look in Models/ and Results/ directories
```

## 10. Notes for Marker

- All scripts run on **CPU by default** (no CUDA required)
- No interactive plots (`matplotlib` set to "Agg" backend)
- Results are saved automatically; no manual export needed
- Early stopping ensures training completes even with high `NUM_EPOCHS`
- Random seeds fixed for reproducibility (SEED = 42)

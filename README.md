# Brain Tumor MRI Classifier

Automated classification system for brain tumor types using deep convolutional neural networks.

## Quick Start

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Prepare the dataset
python split_dataset.py            # split raw images into Dataset/train, Dataset/val, Dataset/test
python data_setup.py              # optional sanity check of the split dataset

# Train models on raw data
python train_baseline.py
python train_improved.py
python MRI_classifier.py

# Visualize AttentionCNN training saliency
python gradcam.py --sequential

# Build ROI dataset and train ROI models
python roicrop.py --input_dir Dataset --output_dir Dataset/ROI
python train_baseline_roi.py
python train_improved_roi.py
python MRI_classifier_roi.py

# Final evaluation and visualization
python evaluate.py
python gradcam.py --all
```

Alternatively, if you have GNU Make installed, use the repository Makefile:

```bash
make setup       # create .venv and install dependencies
make train       # run MRI_classifier.py
make evaluate    # run evaluate.py
make all         # setup, train, and evaluate
```

## Models

| Script | Model | Accuracy |
|--------|-------|----------|
| `MRI_classifier.py` | BrainTumorNet-Lite (SE blocks + MixUp) | TBD |
| `train_baseline.py` | Baseline CNN | TBD |
| `train_improved.py` | Improved CNN | TBD |

## Dataset Structure

```
Dataset/
├── train/ (4 tumor classes)
├── val/   (4 tumor classes)
└── test/  (4 tumor classes)
```

## Output Files

- **Models/** — Trained model checkpoints (.pth)
- **Plots/** — Training curves, confusion matrices, Grad-CAM visualizations
- **Results/** — Classification metrics and test results

## Full Documentation

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for detailed setup, parameter tuning, and troubleshooting.

## System Requirements

- Python 3.9+
- NumPy < 2.0 (important for PyTorch compatibility)
- ~2GB RAM (CPU) or GPU with 4GB+ VRAM
- CPU: ~30-60 min training time | GPU: ~5-10 min

## Key Features

✓ Early stopping (prevents overfitting)  
✓ Test-time augmentation (TTA)  
✓ Grad-CAM saliency visualization  
✓ CPU & GPU support  
✓ Reproducible results (fixed seeds)  
✓ Automatic checkpoint saving

## Troubleshooting

**NumPy compatibility error?**
```bash
pip install "numpy<2"
```

**Out of memory?**
- Reduce `BATCH_SIZE` in the training script
- Use smaller image size

**More info:** See INSTRUCTIONS.md section 8.

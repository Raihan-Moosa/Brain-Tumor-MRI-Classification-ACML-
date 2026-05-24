# Brain Tumor MRI Classifier

Automated classification system for brain tumor types using deep convolutional neural networks.

## Installing uv (Prerequisite)

This project uses `uv` for lightning-fast dependency management. If you do not have `uv` installed, install it first:

* Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
* macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Quick Start

Follow these commands in your terminal to run the pipeline:

* 1. Sync dependencies and create virtual environment: `uv sync`
* 2. Activate the environment (Windows): `.venv\\Scripts\\activate`
* 3. Activate the environment (macOS/Linux): `source .venv/bin/activate`

(Note: Ensure your raw images are already organized inside the `Dataset/` folder before proceeding).

* 4. Train models on raw data:
  * `python train_baseline.py`
  * `python train_improved.py`
  * `python MRI_classifier.py`

* 5. Visualize BrainTumorNet-Lite training saliency (requires models trained above):
  * `python gradcam.py --sequential`

* 6. Build ROI dataset and train ROI models:
  * `python roicrop.py --input_dir Dataset --output_dir Dataset/ROI`
  * `python train_baseline_roi.py`
  * `python train_improved_roi.py`
  * `python MRI_classifier_roi.py`

* 7. Final evaluation and visualization:
  * `python evaluate.py --model all`
  * `python gradcam.py --all`

## Models

| Script | Model | Accuracy |
|--------|-------|----------|
| `MRI_classifier.py` | BrainTumorNet-Lite (SE blocks + MixUp) | TBD |
| `train_baseline.py` | Baseline CNN | TBD |
| `train_improved.py` | Improved CNN | TBD |

## Dataset Structure

Dataset/
├── train/ (4 tumor classes)
├── val/   (4 tumor classes)
└── test/  (4 tumor classes)

## Output Files

- **Models/** — Trained model checkpoints (.pth)
- **Plots/** — Training curves, confusion matrices, Grad-CAM visualizations
- **Results/** — Classification metrics and test results

## Full Documentation

See **INSTRUCTIONS.md** for detailed setup, parameter tuning, and troubleshooting.

## System Requirements

- Python 3.9+
- NumPy < 2.0 (handled automatically by uv)
- ~2GB RAM (CPU) or GPU with 4GB+ VRAM
- CPU: ~30-60 min training time | GPU: ~5-10 min

## Key Features

✓ Early stopping (prevents overfitting)  
✓ Test-time augmentation (TTA)  
✓ Grad-CAM saliency visualization  
✓ CPU & GPU support  
✓ Reproducible results (fixed seeds)  
✓ Automatic checkpoint saving
# Brain Tumor MRI Classification — How to Run

## 1. Environment Setup

This project uses `uv` for fast, reproducible dependency management using the provided `uv.lock` and `pyproject.toml` files.

### Installing uv
If you do not have `uv` installed on your system, install it using the official standalone installer:
* Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
* macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Activating the Environment
Once `uv` is installed, set up the project:
* Create venv and install dependencies: `uv sync`
* Activate on Windows: `.venv\\Scripts\\activate`
* Activate on macOS/Linux: `source .venv/bin/activate`

## 2. Dataset Structure

The dataset must be manually organized as follows before running any scripts:

Dataset/
├── train/
│   ├── glioma/          (training images for glioma tumors)
│   ├── meningioma/      (training images for meningioma tumors)
│   ├── notumor/         (training images with no tumor)
│   └── pituitary/       (training images for pituitary tumors)
├── val/                 (same 4 subdirectories as train)
└── test/                (same 4 subdirectories as train)

## 3. Main Training Scripts

### Option A: Lite Model (Recommended)
* Command: `python MRI_classifier.py`
* Model: BrainTumorNet-Lite (custom CNN with SE blocks, MixUp augmentation, cosine annealing)
* Note: Saves checkpoint frames to `Models/saliency_ckpts/` for sequential evaluation.

### Option B: Improved Model
* Command: `python train_improved.py`
* Model: Standard CNN with BatchNorm and cosine LR

### Option C: Baseline Model
* Command: `python train_baseline.py`
* Model: Simple CNN (no normalisation, fixed LR)

## 4. Grad-CAM Saliency Maps

Once models are trained, run unified visualizations:
* All models: `python gradcam.py --all`
* Specific model: `python gradcam.py --model baseline` (Options: baseline, improved, lite)
* Time-lapse filmstrip: `python gradcam.py --sequential` (Requires MRI_classifier.py to run first)

## 5. Region of Interest (ROI) Pipeline

To focus the models strictly on brain matter:
* 1. Crop images: `python roicrop.py --input_dir Dataset --output_dir Dataset/ROI`
* 2. Train ROI versions: `python train_baseline_roi.py` (and similarly for improved/lite)

## 6. Unified Evaluation

Compare standard vs. ROI models:
* Evaluate all: `python evaluate.py`
* Specific model: `python evaluate.py --model lite --dataset both`

## 7. Troubleshooting

* Q: GPU not detected (CUDA)
  * A: Scripts default to CPU. To use GPU, modify `DEVICE` in main scripts or install CUDA-enabled PyTorch via uv.
* Q: Out of memory on CPU
  * A: Reduce `BATCH_SIZE` (e.g., 16 or 8) in the script.

## 8. Quick Start (30 seconds)

* 1. Sync environment: `uv sync`
* 2. Activate venv: `.venv\\Scripts\\activate`
* 3. Run training: `python MRI_classifier.py`
* 4. Check results: Look in Models/ and Results/ directories
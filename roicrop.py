"""
roicrop.py
====================
Region of Interest (ROI) Masking Pipeline
----------------------------------------------------
Grad-CAM analysis revealed that our classifier exploits peripheral scanning
artifacts (edge-padding, cropping boundaries, skull silhouette) rather than
tumour pathology. This script isolates the biological brain matter so that
every downstream model is forced to learn real morphological signal.

Algorithm (per image)
---------------------
1. Convert to greyscale.
2. Apply Gaussian blur to suppress high-frequency noise before thresholding.
3. Run Otsu's automatic global thresholding – avoids any hand-tuned pixel
   value and is therefore dataset-agnostic.
4. Morphological closing fills small intra-brain holes that Otsu leaves open.
5. Find external contours; keep only the single largest one (the skull/brain
   outline). Discards table labels, scanner frames, and text annotations.
6. Compute the axis-aligned bounding box of that contour and crop.
7. Add a configurable pixel margin so we never accidentally clip sulcal edges.
8. Resize the crop to a fixed output resolution (default 224 × 224) so the
   dataset is immediately compatible with standard CNN input pipelines.

Directory contract
------------------
    INPUT_ROOT/
        glioma/       *.jpg | *.png …
        meningioma/
        pituitary/
        notumor/

    OUTPUT_ROOT/          (mirrors the class structure)
        glioma/
        meningioma/
        pituitary/
        notumor/

Usage
-----
    python roicrop.py \
        --input_dir  Dataset \
        --output_dir Dataset/ROI \ - ONLY THESE 2 OPTIONS STRICTLY NECESSARY
        --output_size 224 \
        --margin 10 \
        --workers 4
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

#---------------------------------------------------------------------------
#Configuration defaults
#---------------------------------------------------------------------------
DEFAULT_OUTPUT_SIZE: int = 224   #pixels – square output
DEFAULT_MARGIN: int = 10         #extra padding around bounding box (pixels)
DEFAULT_BLUR_KSIZE: int = 5      #Gaussian kernel side length (must be odd)
DEFAULT_MORPH_KSIZE: int = 15    #morphological closing kernel side length
VALID_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


#---------------------------------------------------------------------------
#Core per-image processing function
#---------------------------------------------------------------------------

def otsu_threshold(grey: np.ndarray) -> np.ndarray:
    """
    Otsu's method  - produce a binary mask.

    Otsu's algorithm selects the threshold T* that minimises within-class
    variance across the foreground/background pixel distributions:

        T* = argmin_T [ w_bg(T) * σ²_bg(T)  +  w_fg(T) * σ²_fg(T) ]
        
    Returns
    -------
    binary : np.ndarray, dtype=uint8
        Pixel values are 0 (background) or 255 (foreground / brain tissue).
    """
    _, binary = cv2.threshold(
        grey, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def largest_contour_bbox(
    binary: np.ndarray,
    margin: int,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the largest external contour in a binary mask.

    Parameters
    ----------
    binary  : Otsu-thresholded mask.
    margin  : Extra pixels added on all four sides of the tight bounding box.
    img_h   : Original image height.
    img_w   : Original image width.

    Returns
    -------
    (x1, y1, x2, y2) : top-left and bottom-right corners, margin included,
                        clamped to image bounds.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found – image may be entirely black.")

    #Select the contour with the maximum enclosed area
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    #Expand by margin and clamp to valid pixel range
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)

    return x1, y1, x2, y2


def crop_and_resize(
    image_path: Path,
    output_path: Path,
    output_size: int,
    margin: int,
    blur_ksize: int,
    morph_ksize: int,
) -> dict:
    """
    Full ROI crop pipeline for a single MRI image.

    Steps
    -----
    1. Read image in colour (preserves any contrast staining).
    2. Convert a greyscale copy for thresholding.
    3. Gaussian blur  → reduces noise before Otsu.
    4. Otsu threshold → binary foreground mask.
    5. Morphological closing (square structuring element) → fills holes.
    6. Largest contour bounding box with margin.
    7. Crop the *colour* image to that box.
    8. Resize to (output_size × output_size).
    9. Write to output_path.

    Returns
    -------
    dict with keys: 'path', 'status', 'error'
    """
    result = {"path": str(image_path), "status": "ok", "error": None}

    try:
        #--- 1. Read ---
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"cv2.imread returned None for {image_path}")

        img_h, img_w = img_bgr.shape[:2]

        #--- 2. Greyscale copy ---
        grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        #--- 3. Gaussian blur (kernel must be odd × odd) ---
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(grey, (ksize, ksize), sigmaX=0)

        #--- 4. Otsu threshold ---
        binary = otsu_threshold(blurred)

        #--- 5. Morphological closing ---
        #Closing = dilation followed by erosion.
        #Fills small dark regions inside the brain that Otsu misclassifies as
        #background (e.g. ventricles, sulci).
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_ksize, morph_ksize)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        #--- 6. Bounding box of largest contour ---
        x1, y1, x2, y2 = largest_contour_bbox(closed, margin, img_h, img_w)

        #crop must have non-zero area
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Degenerate bounding box ({x1},{y1})→({x2},{y2})."
            )

        #--- 7. Crop colour image ---
        cropped = img_bgr[y1:y2, x1:x2]

        #--- 8. Resize ---
        resized = cv2.resize(
            cropped,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        )

        #--- 9. Write ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), resized)
        if not success:
            raise IOError(f"cv2.imwrite failed for {output_path}")

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result



#Worker shim (needed for ProcessPoolExecutor pickling)
def _worker(args: tuple) -> dict:
    return crop_and_resize(*args)


#--------------------------------------------------------------------------
#Dataset-level orchestrator
#---------------------------------------------------------------------------

def process_dataset(
    input_root: Path,
    output_root: Path,
    output_size: int,
    margin: int,
    blur_ksize: int,
    morph_ksize: int,
    workers: int,
) -> None:
    """
    Walk input_root recursively, collect all valid image files, and process
    them in parallel while mirroring the subdirectory structure under
    output_root.
    """
    #Collect all image paths
    all_images = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not all_images:
        log.error(
            "No images found under '%s' with extensions %s",
            input_root, VALID_EXTENSIONS
        )
        sys.exit(1)

    log.info("Found %d images under '%s'.", len(all_images), input_root)

    #Build (image_path, output_path, …) tuples
    tasks = []
    for img_path in all_images:
        #Reconstruct the same relative sub-path under output_root
        rel = img_path.relative_to(input_root)
        out_path = output_root / rel
        tasks.append((
            img_path, out_path,
            output_size, margin, blur_ksize, morph_ksize,
        ))

    #Process with progress bar
    failed, succeeded = [], []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        with tqdm(total=len(futures), unit="img", desc="ROI cropping") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    succeeded.append(result["path"])
                else:
                    failed.append(result)
                pbar.update(1)

    #Summary report
    log.info("Completed: %d succeeded, %d failed.", len(succeeded), len(failed))

    if failed:
        log.warning("Failed images (first 20 shown):")
        for item in failed[:20]:
            log.warning("  %s  →  %s", item["path"], item["error"])

        #Write full failure log
        fail_log = output_root / "failed_crops.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with fail_log.open("w") as fh:
            for item in failed:
                fh.write(f"{item['path']}\t{item['error']}\n")
        log.info("Full failure log written to '%s'.", fail_log)


#Fancy stuff
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain MRI ROI bounding-box cropping pipeline (Step 6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Root directory of the raw Kaggle dataset.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Destination root for cropped images (mirrors class structure).",
    )
    parser.add_argument(
        "--output_size", type=int, default=DEFAULT_OUTPUT_SIZE,
        help="Side length (px) of the square output image.",
    )
    parser.add_argument(
        "--margin", type=int, default=DEFAULT_MARGIN,
        help="Extra pixels added around the tight bounding box.",
    )
    parser.add_argument(
        "--blur_ksize", type=int, default=DEFAULT_BLUR_KSIZE,
        help="Gaussian blur kernel size (must be odd).",
    )
    parser.add_argument(
        "--morph_ksize", type=int, default=DEFAULT_MORPH_KSIZE,
        help="Morphological closing kernel size.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input_dir.exists():
        log.error("Input directory '%s' does not exist.", args.input_dir)
        sys.exit(1)

    log.info("Input  : %s", args.input_dir)
    log.info("Output : %s", args.output_dir)
    log.info(
        "Settings: size=%dpx  margin=%dpx  blur=%d  morph=%d  workers=%d",
        args.output_size, args.margin,
        args.blur_ksize, args.morph_ksize, args.workers,
    )

    process_dataset(
        input_root=args.input_dir,
        output_root=args.output_dir,
        output_size=args.output_size,
        margin=args.margin,
        blur_ksize=args.blur_ksize,
        morph_ksize=args.morph_ksize,
        workers=args.workers,
    )

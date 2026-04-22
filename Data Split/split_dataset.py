import random
import shutil
from pathlib import Path

random.seed(42)

combined_dir = Path(r"C:\Users\raiha\Downloads\MRI Tumor Dataset\Combined")
output_dir = Path(r"C:\Users\raiha\Downloads\MRI Tumor Dataset\split_70_15_15")

classes = ["glioma", "meningioma", "notumor", "pituitary"]

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Create output folders
for split in ["train", "val", "test"]:
    for cls in classes:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

# Split each class separately
for cls in classes:
    class_dir = combined_dir / cls
    files = [f for f in class_dir.iterdir() if f.is_file()]
    random.shuffle(files)

    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    for f in train_files:
        shutil.copy2(f, output_dir / "train" / cls / f.name)

    for f in val_files:
        shutil.copy2(f, output_dir / "val" / cls / f.name)

    for f in test_files:
        shutil.copy2(f, output_dir / "test" / cls / f.name)

    print(f"{cls}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Done.")
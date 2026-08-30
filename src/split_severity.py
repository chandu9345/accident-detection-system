import os
import shutil
import glob

TRAIN_SRC = "data/accident_dataset/train/accident"
VAL_SRC   = "data/accident_dataset/val/accident"
TRAIN_DST = "data/severity_dataset/train"
VAL_DST   = "data/severity_dataset/val"
CLASSES   = ["Minor", "Moderate", "Severe"]

def split_and_copy(src_folder, dst_folder):
    for cls in CLASSES:
        os.makedirs(os.path.join(dst_folder, cls), exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(src_folder, "*.jpg")) +
        glob.glob(os.path.join(src_folder, "*.jpeg")) +
        glob.glob(os.path.join(src_folder, "*.png"))
    )

    total = len(images)
    print(f"Found {total} images in {src_folder}")

    third = total // 3
    splits = {
        "Minor"   : images[:third],
        "Moderate": images[third:third*2],
        "Severe"  : images[third*2:]
    }

    for cls, imgs in splits.items():
        for img_path in imgs:
            dst_path = os.path.join(dst_folder, cls, os.path.basename(img_path))
            shutil.copy2(img_path, dst_path)
        print(f"  {cls}: {len(imgs)} images copied")

print("=== Splitting Train ===")
split_and_copy(TRAIN_SRC, TRAIN_DST)

print("\n=== Splitting Val ===")
split_and_copy(VAL_SRC, VAL_DST)

print("\n✅ Done! Severity dataset ready at data/severity_dataset/")
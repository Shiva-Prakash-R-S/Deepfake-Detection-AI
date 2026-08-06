import os
import random
import shutil

# ----------------------------
# Paths
# ----------------------------
SOURCE = "dataset/real_vs_fake/real-vs-fake"
DEST = "dataset/real_vs_fake_small/real-vs-fake"

# Number of images to copy
LIMITS = {
    "train": 5000,
    "valid": 1000,
    "test": 1000
}

CLASSES = ["fake", "real"]

random.seed(42)  # Reproducible results

for split in LIMITS:

    for cls in CLASSES:

        src_folder = os.path.join(SOURCE, split, cls)
        dst_folder = os.path.join(DEST, split, cls)

        os.makedirs(dst_folder, exist_ok=True)

        images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        selected = random.sample(images, LIMITS[split])

        print(f"Copying {len(selected)} {cls} images for {split}...")

        for img in selected:
            shutil.copy2(
                os.path.join(src_folder, img),
                os.path.join(dst_folder, img)
            )

print("\nDataset created successfully!")
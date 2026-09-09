import os
import random
import shutil

from PIL import Image
from tqdm import tqdm

from utils.config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    EXTRACTED_TRAIN_DIR,
    EXTRACTED_VALID_DIR,
    EXTRACTED_TEST_DIR,
)

from utils.face_crop import crop_face


# ============================================================
# SETTINGS
# ============================================================

MAX_TRAIN_PER_CLASS = 15000
MAX_VALID_PER_CLASS = 2000
MAX_TEST_PER_CLASS = 2000

RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ============================================================
# SELECT IMAGES
# ============================================================

def get_images(folder, max_images):

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"\nFolder not found:\n{folder}"
        )

    files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.seed(RANDOM_SEED)

    if len(files) > max_images:

        files = random.sample(
            files,
            max_images
        )

    return files


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    source_dir,
    output_dir,
    max_per_class
):

    print("\n")
    print("=" * 60)
    print("Processing:")
    print(source_dir)
    print("=" * 60)

    total = 0
    detected = 0
    skipped = 0

    for class_name in ["fake", "real"]:

        source_class_dir = os.path.join(
            source_dir,
            class_name
        )

        output_class_dir = os.path.join(
            output_dir,
            class_name
        )

        # ----------------------------------------------------
        # Check source class folder
        # ----------------------------------------------------

        if not os.path.exists(source_class_dir):

            raise FileNotFoundError(
                f"\nClass folder not found:\n"
                f"{source_class_dir}"
            )

        os.makedirs(
            output_class_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Select images
        # ----------------------------------------------------

        files = get_images(
            source_class_dir,
            max_per_class
        )

        print(
            f"\n{class_name.upper()}"
        )

        print(
            f"Selected images: {len(files)}"
        )

        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        for filename in tqdm(
            files,
            desc=f"{class_name} face detection"
        ):

            total += 1

            source_path = os.path.join(
                source_class_dir,
                filename
            )

            try:

                # ------------------------------------------------
                # Open image
                # ------------------------------------------------

                image = Image.open(
                    source_path
                ).convert("RGB")

                # ------------------------------------------------
                # Detect and crop face
                # ------------------------------------------------

                face = crop_face(
                    image
                )

                # ------------------------------------------------
                # No face detected
                # ------------------------------------------------

                if face is None:

                    skipped += 1

                    continue

                # ------------------------------------------------
                # Output filename
                # ------------------------------------------------

                base_name = os.path.splitext(
                    filename
                )[0]

                output_path = os.path.join(
                    output_class_dir,
                    base_name + ".jpg"
                )

                # ------------------------------------------------
                # Save face crop
                # ------------------------------------------------

                face.save(
                    output_path,
                    "JPEG",
                    quality=95
                )

                detected += 1

            except Exception as error:

                skipped += 1

                print(
                    f"\n⚠ Error processing: {filename}"
                )

                print(
                    f"   {error}"
                )

    # ========================================================
    # SPLIT RESULTS
    # ========================================================

    print("\n")
    print("-" * 60)

    print(
        f"Total selected : {total}"
    )

    print(
        f"Faces detected : {detected}"
    )

    print(
        f"Skipped        : {skipped}"
    )

    print("-" * 60)

    return (
        total,
        detected,
        skipped
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("      V3 FACE PREPROCESSING")
    print("=" * 60)

    print("\nDataset:")
    print(
        TRAIN_DIR
        .replace("\\train", "")
    )

    print("\nLimits:")

    print(
        f"Train      : "
        f"{MAX_TRAIN_PER_CLASS} per class"
    )

    print(
        f"Validation : "
        f"{MAX_VALID_PER_CLASS} per class"
    )

    print(
        f"Test       : "
        f"{MAX_TEST_PER_CLASS} per class"
    )

    print("=" * 60)

    # ========================================================
    # CHECK ORIGINAL DATASET
    # ========================================================

    print("\nChecking dataset directories...")

    for directory in [
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    ]:

        if not os.path.exists(directory):

            raise FileNotFoundError(
                f"\nDataset directory not found:\n"
                f"{directory}"
            )

        print(
            f"✅ {directory}"
        )

    # ========================================================
    # REMOVE OLD EXTRACTED DATA
    # ========================================================

    extracted_root = os.path.dirname(
        EXTRACTED_TRAIN_DIR
    )

    if os.path.exists(
        extracted_root
    ):

        print(
            "\n⚠ Removing previous extracted_faces..."
        )

        shutil.rmtree(
            extracted_root
        )

        print(
            "✅ Old extracted dataset removed."
        )

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    os.makedirs(
        EXTRACTED_TRAIN_DIR,
        exist_ok=True
    )

    os.makedirs(
        EXTRACTED_VALID_DIR,
        exist_ok=True
    )

    os.makedirs(
        EXTRACTED_TEST_DIR,
        exist_ok=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_total, train_detected, train_skipped = process_split(
        TRAIN_DIR,
        EXTRACTED_TRAIN_DIR,
        MAX_TRAIN_PER_CLASS
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    valid_total, valid_detected, valid_skipped = process_split(
        VALID_DIR,
        EXTRACTED_VALID_DIR,
        MAX_VALID_PER_CLASS
    )

    # ========================================================
    # TEST
    # ========================================================

    test_total, test_detected, test_skipped = process_split(
        TEST_DIR,
        EXTRACTED_TEST_DIR,
        MAX_TEST_PER_CLASS
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 60)
    print("    FACE PREPROCESSING COMPLETED")
    print("=" * 60)

    print("\nTRAIN")
    print(
        f"Selected : {train_total}"
    )
    print(
        f"Faces    : {train_detected}"
    )
    print(
        f"Skipped  : {train_skipped}"
    )

    print("\nVALIDATION")
    print(
        f"Selected : {valid_total}"
    )
    print(
        f"Faces    : {valid_detected}"
    )
    print(
        f"Skipped  : {valid_skipped}"
    )

    print("\nTEST")
    print(
        f"Selected : {test_total}"
    )
    print(
        f"Faces    : {test_detected}"
    )
    print(
        f"Skipped  : {test_skipped}"
    )

    print("\n")
    print("=" * 60)

    print(
        "Face-cropped dataset created at:"
    )

    print(
        EXTRACTED_TRAIN_DIR
    )

    print(
        EXTRACTED_VALID_DIR
    )

    print(
        EXTRACTED_TEST_DIR
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
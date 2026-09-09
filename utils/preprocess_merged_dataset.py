import os
import random
import shutil

from PIL import Image
from tqdm import tqdm

from utils.face_crop import crop_face


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = os.path.join(
    "dataset",
    "merged_fake_real"
)

OUTPUT_DIR = os.path.join(
    "dataset",
    "merged_extracted_faces"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_TRAIN_PER_CLASS = 24000
MAX_VALID_PER_CLASS = 3000
MAX_TEST_PER_CLASS = 3000

RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder, max_images):

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    random.seed(RANDOM_SEED)

    if len(files) > max_images:

        files = random.sample(
            files,
            max_images
        )

    return files


# ============================================================
# PROCESS SPLIT
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

        os.makedirs(
            output_class_dir,
            exist_ok=True
        )

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

                image = Image.open(
                    source_path
                ).convert("RGB")

                face = crop_face(
                    image
                )

                if face is None:

                    skipped += 1
                    continue

                base_name = os.path.splitext(
                    filename
                )[0]

                output_path = os.path.join(
                    output_class_dir,
                    base_name + ".jpg"
                )

                # Avoid overwriting if the same filename
                # somehow occurs.
                counter = 1

                while os.path.exists(
                    output_path
                ):

                    output_path = os.path.join(
                        output_class_dir,
                        f"{base_name}_{counter}.jpg"
                    )

                    counter += 1

                face.save(
                    output_path,
                    "JPEG",
                    quality=95
                )

                detected += 1

            except Exception as error:

                skipped += 1

                print(
                    f"\n⚠ Error: {filename}"
                )

                print(error)

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       V4 FACE PREPROCESSING")
    print("=" * 60)

    print("\nSource:")
    print(
        os.path.abspath(SOURCE_DIR)
    )

    print("\nOutput:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print("\nLimits:")
    print(
        f"Train      : {MAX_TRAIN_PER_CLASS} per class"
    )

    print(
        f"Validation : {MAX_VALID_PER_CLASS} per class"
    )

    print(
        f"Test       : {MAX_TEST_PER_CLASS} per class"
    )

    print("=" * 60)


    # ========================================================
    # CHECK SOURCE
    # ========================================================

    print("\nChecking source dataset...")

    for split in [
        "train",
        "valid",
        "test"
    ]:

        for class_name in [
            "fake",
            "real"
        ]:

            folder = os.path.join(
                SOURCE_DIR,
                split,
                class_name
            )

            if not os.path.exists(folder):

                raise FileNotFoundError(
                    f"\nMissing folder:\n{folder}"
                )

            print(
                "✅",
                folder
            )


    # ========================================================
    # REMOVE ONLY OLD V4 OUTPUT
    # ========================================================

    if os.path.exists(
        OUTPUT_DIR
    ):

        print(
            "\n⚠ Removing previous V4 extracted dataset..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )


    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # TRAIN
    # ========================================================

    process_split(
        os.path.join(
            SOURCE_DIR,
            "train"
        ),
        os.path.join(
            OUTPUT_DIR,
            "train"
        ),
        MAX_TRAIN_PER_CLASS
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    process_split(
        os.path.join(
            SOURCE_DIR,
            "valid"
        ),
        os.path.join(
            OUTPUT_DIR,
            "valid"
        ),
        MAX_VALID_PER_CLASS
    )


    # ========================================================
    # TEST
    # ========================================================

    process_split(
        os.path.join(
            SOURCE_DIR,
            "test"
        ),
        os.path.join(
            OUTPUT_DIR,
            "test"
        ),
        MAX_TEST_PER_CLASS
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 60)
    print("    V4 FACE PREPROCESSING COMPLETED")
    print("=" * 60)

    print("\nOutput:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print("\nNext step:")
    print(
        "Update utils/config.py and dataset_loader.py "
        "to use merged_extracted_faces."
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
import os
import shutil
import random
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Existing Real-vs-Fake dataset
REAL_VS_FAKE_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "real_vs_fake",
    "real-vs-fake"
)

# AI-vs-Human dataset
AI_TRAIN_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "train_data"
)

AI_TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "dataset",
    "train.csv"
)

# New merged dataset
MERGED_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "merged_fake_real"
)

# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Per class
REAL_VS_FAKE_COUNT = 12000
AI_VS_HUMAN_COUNT = 18000

TOTAL_PER_CLASS = (
    REAL_VS_FAKE_COUNT +
    AI_VS_HUMAN_COUNT
)

# Validation and test percentage
VALID_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(RANDOM_SEED)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    for split in ["train", "valid", "test"]:

        for class_name in ["fake", "real"]:

            path = os.path.join(
                MERGED_DIR,
                split,
                class_name
            )

            os.makedirs(
                path,
                exist_ok=True
            )


# ============================================================
# CLEAR OLD MERGED DATASET
# ============================================================

def clear_old_dataset():

    if os.path.exists(MERGED_DIR):

        print("\nRemoving previous merged dataset...")

        shutil.rmtree(MERGED_DIR)

    create_directories()


# ============================================================
# GET REAL-VS-FAKE IMAGES
# ============================================================

def get_real_vs_fake_images():

    selected = {
        "fake": [],
        "real": []
    }

    for class_name in ["fake", "real"]:

        folder = os.path.join(
            REAL_VS_FAKE_DIR,
            "train",
            class_name
        )

        if not os.path.exists(folder):

            raise FileNotFoundError(
                f"Folder not found:\n{folder}"
            )

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            )
        ]

        if len(files) < REAL_VS_FAKE_COUNT:

            raise ValueError(
                f"Not enough {class_name} images.\n"
                f"Required: {REAL_VS_FAKE_COUNT}\n"
                f"Available: {len(files)}"
            )

        random.shuffle(files)

        selected[class_name] = files[
            :REAL_VS_FAKE_COUNT
        ]

        print(
            f"Real-vs-Fake {class_name}: "
            f"{len(selected[class_name])}"
        )

    return selected


# ============================================================
# GET AI-VS-HUMAN IMAGES
# ============================================================

def get_ai_vs_human_images():

    if not os.path.exists(AI_TRAIN_CSV):

        raise FileNotFoundError(
            f"CSV not found:\n{AI_TRAIN_CSV}"
        )

    df = pd.read_csv(
        AI_TRAIN_CSV
    )

    # Label mapping:
    # 0 = Human / Real
    # 1 = AI-generated / Fake

    selected = {
        "fake": [],
        "real": []
    }

    for label, class_name in [
        (1, "fake"),
        (0, "real")
    ]:

        rows = df[
            df["label"] == label
        ]

        if len(rows) < AI_VS_HUMAN_COUNT:

            raise ValueError(
                f"Not enough AI-vs-Human "
                f"{class_name} images.\n"
                f"Required: {AI_VS_HUMAN_COUNT}\n"
                f"Available: {len(rows)}"
            )

        rows = rows.sample(
            n=AI_VS_HUMAN_COUNT,
            random_state=RANDOM_SEED
        )

        for relative_path in rows["file_name"]:

            source_path = os.path.join(
                PROJECT_DIR,
                "dataset",
                relative_path
            )

            if os.path.exists(source_path):

                selected[class_name].append(
                    source_path
                )

    print(
        f"AI-vs-Human fake: "
        f"{len(selected['fake'])}"
    )

    print(
        f"AI-vs-Human real: "
        f"{len(selected['real'])}"
    )

    return selected


# ============================================================
# SPLIT IMAGES
# ============================================================

def split_images(images):

    random.shuffle(images)

    total = len(images)

    test_count = int(
        total * TEST_RATIO
    )

    valid_count = int(
        total * VALID_RATIO
    )

    test_images = images[
        :test_count
    ]

    valid_images = images[
        test_count:
        test_count + valid_count
    ]

    train_images = images[
        test_count + valid_count:
    ]

    return (
        train_images,
        valid_images,
        test_images
    )


# ============================================================
# COPY IMAGE
# ============================================================

def copy_image(
    source,
    destination,
    prefix,
    index
):

    extension = os.path.splitext(
        source
    )[1].lower()

    filename = (
        f"{prefix}_{index:06d}{extension}"
    )

    destination_path = os.path.join(
        destination,
        filename
    )

    shutil.copy2(
        source,
        destination_path
    )


# ============================================================
# BUILD CLASS
# ============================================================

def build_class(
    class_name,
    real_vs_fake_images,
    ai_vs_human_images
):

    print("\n" + "=" * 60)

    print(
        f"BUILDING CLASS: {class_name.upper()}"
    )

    print("=" * 60)

    combined = []

    # Existing dataset
    for image in real_vs_fake_images:

        combined.append(
            ("rvf", image)
        )

    # AI-vs-Human dataset
    for image in ai_vs_human_images:

        combined.append(
            ("aih", image)
        )

    random.shuffle(combined)

    # Total = 30,000
    print(
        f"Total images: {len(combined)}"
    )

    # Split
    train_count = int(
        len(combined) * 0.80
    )

    valid_count = int(
        len(combined) * 0.10
    )

    train_data = combined[
        :train_count
    ]

    valid_data = combined[
        train_count:
        train_count + valid_count
    ]

    test_data = combined[
        train_count + valid_count:
    ]

    splits = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }

    for split_name, data in splits.items():

        destination = os.path.join(
            MERGED_DIR,
            split_name,
            class_name
        )

        print(
            f"{split_name}: {len(data)}"
        )

        for index, (prefix, source) in enumerate(data):

            copy_image(
                source,
                destination,
                f"{class_name}_{prefix}",
                index
            )

    print(
        f"✅ {class_name.upper()} completed"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       MERGED DATASET CREATION")
    print("=" * 60)

    print(
        "\nReal-vs-Fake per class:",
        REAL_VS_FAKE_COUNT
    )

    print(
        "AI-vs-Human per class:",
        AI_VS_HUMAN_COUNT
    )

    print(
        "Total per class:",
        TOTAL_PER_CLASS
    )

    print(
        "\nRatio:"
    )

    print(
        "Real-vs-Fake = 40%"
    )

    print(
        "AI-vs-Human  = 60%"
    )

    # --------------------------------------------------------
    # Clear old dataset
    # --------------------------------------------------------

    clear_old_dataset()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading Real-vs-Fake dataset...")

    rvf = get_real_vs_fake_images()

    print("\nLoading AI-vs-Human dataset...")

    aih = get_ai_vs_human_images()

    # --------------------------------------------------------
    # Build Fake
    # --------------------------------------------------------

    build_class(
        "fake",
        rvf["fake"],
        aih["fake"]
    )

    # --------------------------------------------------------
    # Build Real
    # --------------------------------------------------------

    build_class(
        "real",
        rvf["real"],
        aih["real"]
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("       MERGED DATASET CREATED")
    print("=" * 60)

    print(
        "\nLocation:"
    )

    print(
        MERGED_DIR
    )

    print("\nStructure:")

    print(
        "train/fake : 24,000"
    )

    print(
        "train/real : 24,000"
    )

    print(
        "valid/fake : 3,000"
    )

    print(
        "valid/real : 3,000"
    )

    print(
        "test/fake  : 3,000"
    )

    print(
        "test/real  : 3,000"
    )

    print("\n✅ Dataset merge completed!")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
import io
import os
import random

from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.config import (
    EXTRACTED_TRAIN_DIR,
    EXTRACTED_VALID_DIR,
    EXTRACTED_TEST_DIR,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    BATCH_SIZE,
    NUM_WORKERS
)


# ============================================================
# JPEG COMPRESSION AUGMENTATION
# ============================================================

class RandomJPEGCompression:

    def __init__(
        self,
        quality_range=(30, 90),
        probability=0.4
    ):

        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image):

        if random.random() > self.probability:
            return image

        quality = random.randint(
            self.quality_range[0],
            self.quality_range[1]
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        return Image.open(
            buffer
        ).convert("RGB")


# ============================================================
# TRAIN TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3
            )
        ],
        p=0.3
    ),

    RandomJPEGCompression(
        quality_range=(30, 90),
        probability=0.4
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORM_MEAN,
        std=NORM_STD
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORM_MEAN,
        std=NORM_STD
    )
])


# ============================================================
# CHECK V4 DATASET
# ============================================================

def check_dataset():

    directories = [
        EXTRACTED_TRAIN_DIR,
        EXTRACTED_VALID_DIR,
        EXTRACTED_TEST_DIR
    ]

    print("\n" + "=" * 60)
    print("              V4 DATASET CHECK")
    print("=" * 60)

    for directory in directories:

        if not os.path.exists(directory):

            raise FileNotFoundError(
                f"\nV4 face dataset not found:\n"
                f"{directory}\n\n"
                f"Run:\n"
                f"python -m utils.preprocess_merged_dataset"
            )

        print(
            "✅",
            directory
        )

    print("=" * 60)


# ============================================================
# BUILD LOADERS
# ============================================================

def build_loaders():

    check_dataset()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        EXTRACTED_TRAIN_DIR,
        transform=train_transform
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_dataset = datasets.ImageFolder(
        EXTRACTED_VALID_DIR,
        transform=test_transform
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        EXTRACTED_TEST_DIR,
        transform=test_transform
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("              V4 DATASET INFORMATION")
    print("=" * 60)

    print(
        "\nClasses:",
        train_dataset.classes
    )

    print(
        "Training Images   :",
        len(train_dataset)
    )

    print(
        "Validation Images :",
        len(valid_dataset)
    )

    print(
        "Testing Images    :",
        len(test_dataset)
    )

    print(
        "Batch Size        :",
        BATCH_SIZE
    )

    print(
        "Workers           :",
        NUM_WORKERS
    )

    print("=" * 60)

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset.classes
    )


# ============================================================
# GLOBAL LOADERS
# ============================================================

(
    train_loader,
    valid_loader,
    test_loader,
    classes
) = build_loaders()
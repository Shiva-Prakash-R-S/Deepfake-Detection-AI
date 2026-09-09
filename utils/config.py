import os

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

FACE_MARGIN = 40

NORM_MEAN = [
    0.485,
    0.456,
    0.406
]

NORM_STD = [
    0.229,
    0.224,
    0.225
]

# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "fake",
    "real"
]

# ============================================================
# ORIGINAL DATASET
# ============================================================

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "real_vs_fake",
    "real-vs-fake"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VALID_DIR = os.path.join(
    DATASET_DIR,
    "valid"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

# ============================================================
# V4 MERGED DATASET
# ============================================================

MERGED_DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "merged_fake_real"
)

# ============================================================
# V4 FACE-CROPPED DATASET
# ============================================================

MERGED_EXTRACTED_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "merged_extracted_faces"
)

EXTRACTED_TRAIN_DIR = os.path.join(
    MERGED_EXTRACTED_DIR,
    "train"
)

EXTRACTED_VALID_DIR = os.path.join(
    MERGED_EXTRACTED_DIR,
    "valid"
)

EXTRACTED_TEST_DIR = os.path.join(
    MERGED_EXTRACTED_DIR,
    "test"
)

# ============================================================
# MODEL
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

# V3 model - starting checkpoint
V3_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "best_model_v3.pth"
)

# V2 model - previous checkpoint
V2_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "best_model_v2.pth"
)

# V4 model - will be created after training
V4_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "best_model_v4.pth"
)

# Model used by evaluation/prediction
CURRENT_CHECKPOINT = V4_CHECKPOINT

# ============================================================
# REPORTS
# ============================================================

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

V4_REPORTS_DIR = os.path.join(
    REPORTS_DIR,
    "v4"
)

# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5
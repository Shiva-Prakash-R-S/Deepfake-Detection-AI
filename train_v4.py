import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm

from backend.models.deepfake_model import DeepfakeDetector

from utils.dataset_loader import (
    train_loader,
    valid_loader
)

from utils.config import (
    V3_CHECKPOINT,
    V4_CHECKPOINT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REPORTS_DIR
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device :", device)


# ============================================================
# MODEL
# ============================================================

model = DeepfakeDetector().to(device)


# ============================================================
# LOAD V3 CHECKPOINT
# ============================================================

if not os.path.exists(V3_CHECKPOINT):

    raise FileNotFoundError(
        f"\nV3 checkpoint not found:\n"
        f"{V3_CHECKPOINT}"
    )


checkpoint = torch.load(
    V3_CHECKPOINT,
    map_location=device,
    weights_only=False
)

# Support both plain state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


print("\n✅ Previous V3 model loaded:")
print(V3_CHECKPOINT)
print("V4 fine-tuning will continue from V3.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(V4_CHECKPOINT),
    exist_ok=True
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    REPORTS_DIR,
    "training_history_v4.csv"
)

history = []


# ============================================================
# BEST VALUES
# ============================================================

best_validation_accuracy = 0.0
best_validation_loss = float("inf")

epochs_without_improvement = 0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)

    print(
        f"V4 EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss_total = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            validation_loss_total += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += (
                labels.size(0)
            )


    validation_loss = (
        validation_loss_total /
        validation_total
    )

    validation_accuracy = (
        validation_correct /
        validation_total
    ) * 100


    # ========================================================
    # RESULTS
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n## Results")

    print(
        f"Train Loss          : "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss     : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Train Accuracy      : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy : "
        f"{validation_accuracy:.2f}%"
    )

    print(
        f"Learning Rate       : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history.append([
        epoch + 1,
        train_loss,
        validation_loss,
        train_accuracy,
        validation_accuracy,
        current_lr
    ])


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        best_validation_loss = (
            validation_loss
        )

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            V4_CHECKPOINT
        )

        print(
            "\n✅ BEST V4 MODEL SAVED!"
        )

        print(
            f"Validation Accuracy : "
            f"{validation_accuracy:.2f}%"
        )

        print(
            f"Model : "
            f"{V4_CHECKPOINT}"
        )

    else:

        epochs_without_improvement += 1

        print(
            "\nNo validation improvement."
        )

        print(
            f"Patience: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\n⚠ Early stopping triggered."
        )

        break


# ============================================================
# SAVE CSV
# ============================================================

with open(
    history_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "learning_rate"
    ])

    writer.writerows(
        history
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)

print(
    "       V4 TRAINING COMPLETED"
)

print("=" * 60)

print(
    f"\nBest Validation Accuracy : "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"\nBest Model:"
)

print(
    V4_CHECKPOINT
)

print(
    "\nTraining History:"
)

print(
    history_path
)

print("=" * 60)
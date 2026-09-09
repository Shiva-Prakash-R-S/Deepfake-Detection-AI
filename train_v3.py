import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.deepfake_model import DeepfakeDetector
from utils.dataset_loader import train_loader, valid_loader
from utils.config import (
    CURRENT_CHECKPOINT,
    PREVIOUS_CHECKPOINT,
    REPORTS_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    os.path.dirname(CURRENT_CHECKPOINT),
    exist_ok=True
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n")
print("=" * 60)
print("        DEEPFAKE DETECTION V3 TRAINING")
print("=" * 60)

print(
    f"Using Device : {device}"
)


# ============================================================
# MODEL
# ============================================================

model = DeepfakeDetector().to(device)


# ============================================================
# LOAD V2 MODEL IF AVAILABLE
# ============================================================

if os.path.exists(PREVIOUS_CHECKPOINT):

    try:

        model.load_state_dict(
            torch.load(
                PREVIOUS_CHECKPOINT,
                map_location=device
            )
        )

        print(
            "\n✅ Previous V2 model loaded:"
        )

        print(
            PREVIOUS_CHECKPOINT
        )

        print(
            "V3 fine-tuning will continue from V2."
        )

    except Exception as error:

        print(
            "\n⚠ V2 model could not be loaded."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Training V3 from scratch."
        )

else:

    print(
        "\n⚠ No V2 model found."
    )

    print(
        "Training V3 from scratch."
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_accuracy = 0.0

early_stop_counter = 0

history = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    train_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in train_bar:

        images = images.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        train_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_total += labels.size(0)


        train_correct += (
            predictions == labels
        ).sum().item()


        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    avg_train_loss = (
        train_loss /
        len(train_loader)
    )


    train_accuracy = (
        100.0 *
        train_correct /
        train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            validation_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            validation_total += labels.size(0)


            validation_correct += (
                predictions == labels
            ).sum().item()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    avg_validation_loss = (
        validation_loss /
        len(valid_loader)
    )


    validation_accuracy = (
        100.0 *
        validation_correct /
        validation_total
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(
        validation_accuracy
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history.append({

        "epoch": epoch + 1,

        "train_loss":
            avg_train_loss,

        "validation_loss":
            avg_validation_loss,

        "train_accuracy":
            train_accuracy,

        "validation_accuracy":
            validation_accuracy,

        "learning_rate":
            current_lr
    })


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("Results")
    print("-" * 60)

    print(
        f"Train Loss          : "
        f"{avg_train_loss:.4f}"
    )

    print(
        f"Validation Loss     : "
        f"{avg_validation_loss:.4f}"
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
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        early_stop_counter = 0


        torch.save(
            model.state_dict(),
            CURRENT_CHECKPOINT
        )


        print("\n")
        print("✅ BEST V3 MODEL SAVED!")

        print(
            f"Validation Accuracy : "
            f"{best_accuracy:.2f}%"
        )

        print(
            f"Model : {CURRENT_CHECKPOINT}"
        )


    else:

        early_stop_counter += 1


        print(
            f"\n⚠ No Improvement "
            f"({early_stop_counter}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            early_stop_counter >=
            EARLY_STOPPING_PATIENCE
        ):

            print("\n")
            print(
                "🛑 Early Stopping Triggered!"
            )

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = os.path.join(
    REPORTS_DIR,
    "training_history_v3.csv"
)


with open(
    history_file,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "validation_loss",
            "train_accuracy",
            "validation_accuracy",
            "learning_rate"
        ]
    )

    writer.writeheader()

    writer.writerows(
        history
    )


# ============================================================
# CREATE GRAPHS
# ============================================================

epochs_completed = [
    row["epoch"]
    for row in history
]


train_losses = [
    row["train_loss"]
    for row in history
]


validation_losses = [
    row["validation_loss"]
    for row in history
]


train_accuracies = [
    row["train_accuracy"]
    for row in history
]


validation_accuracies = [
    row["validation_accuracy"]
    for row in history
]


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_completed,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs_completed,
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "V3 Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "loss_v3.png"
    )
)

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_completed,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "V3 Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "accuracy_v3.png"
    )
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("           V3 TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_accuracy:.2f}%"
)

print(
    f"\nBest Model:"
)

print(
    CURRENT_CHECKPOINT
)

print(
    "\nTraining History:"
)

print(
    history_file
)

print(
    "\nGraphs:"
)

print(
    os.path.join(
        REPORTS_DIR,
        "loss_v3.png"
    )
)

print(
    os.path.join(
        REPORTS_DIR,
        "accuracy_v3.png"
    )
)

print("=" * 60)
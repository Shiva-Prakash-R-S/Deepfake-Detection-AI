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

# ===========================================
# Main Function
# ===========================================

def main():

    # ---------------------------------------
    # Create folders
    # ---------------------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ---------------------------------------
    # Device
    # ---------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 60)
    print("      DEEPFAKE DETECTION TRAINING")
    print("=" * 60)
    print(f"Using Device : {device}")

    # ---------------------------------------
    # Model
    # ---------------------------------------
    model = DeepfakeDetector().to(device)

    # ---------------------------------------
    # Load Previous Model
    # ---------------------------------------
    checkpoint = "models/best_model_v1.pth"

    if os.path.exists(checkpoint):

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device
            )
        )

        print("Loaded : best_model_v1.pth")

    else:

        print("No previous model found.")
        print("Training from scratch...")

    # ---------------------------------------
    # Loss
    # ---------------------------------------
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------
    # Optimizer
    # ---------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )

    # ---------------------------------------
    # Scheduler
    # ---------------------------------------
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # ---------------------------------------
    # Settings
    # ---------------------------------------
    EPOCHS = 20

    best_accuracy = 0

    patience = 5

    counter = 0

    train_losses = []
    valid_losses = []

    train_acc_list = []
    valid_acc_list = []

    csv_file = "reports/training_history.csv"

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Validation Loss",
            "Train Accuracy",
            "Validation Accuracy"
        ])
            # ===========================================
    # Training Loop
    # ===========================================

    for epoch in range(EPOCHS):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        # -------------------------------
        # Training
        # -------------------------------
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        train_bar = tqdm(
            train_loader,
            desc="Training",
            leave=True
        )

        for images, labels in train_bar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            running_total += labels.size(0)

            running_correct += (predicted == labels).sum().item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / len(train_loader)

        train_accuracy = (
            100 * running_correct / running_total
        )

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()

        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():

            valid_bar = tqdm(
                valid_loader,
                desc="Validation",
                leave=False
            )

            for images, labels in valid_bar:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                valid_total += labels.size(0)

                valid_correct += (
                    predicted == labels
                ).sum().item()

        valid_loss /= len(valid_loader)

        valid_accuracy = (
            100 * valid_correct / valid_total
        )

        # -------------------------------
        # Scheduler
        # -------------------------------
        scheduler.step(valid_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_acc_list.append(train_accuracy)
        valid_acc_list.append(valid_accuracy)

        # -------------------------------
        # Save CSV
        # -------------------------------
        with open(csv_file, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                epoch + 1,
                round(train_loss, 4),
                round(valid_loss, 4),
                round(train_accuracy, 2),
                round(valid_accuracy, 2)
            ])

        # -------------------------------
        # Print Results
        # -------------------------------
        print("\nResults")
        print("-" * 60)
        print(f"Train Loss        : {train_loss:.4f}")
        print(f"Validation Loss   : {valid_loss:.4f}")
        print(f"Train Accuracy    : {train_accuracy:.2f}%")
        print(f"Validation Accuracy : {valid_accuracy:.2f}%")
        print(f"Learning Rate     : {current_lr:.8f}")
                # -------------------------------
        # Save Best Model
        # -------------------------------
        if valid_accuracy > best_accuracy:

            best_accuracy = valid_accuracy

            counter = 0

            torch.save(
                model.state_dict(),
                "models/best_model_v2.pth"
            )

            print("\n✅ Best Model Saved!")
            print(f"Validation Accuracy : {best_accuracy:.2f}%")

        else:

            counter += 1

            print(f"\n⚠ No Improvement ({counter}/{patience})")

            if counter >= patience:

                print("\n🛑 Early Stopping Triggered!")
                break

    # ===========================================
    # Training Completed
    # ===========================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    print("Model Saved : models/best_model_v2.pth")

    print("Training History : reports/training_history.csv")

    print("=" * 60)

    # ===========================================
    # Plot Loss Graph
    # ===========================================

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        train_losses,
        marker='o',
        label="Train Loss"
    )

    plt.plot(
        epochs,
        valid_losses,
        marker='o',
        label="Validation Loss"
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.savefig("reports/loss_curve.png")

    plt.close()

    print("✅ Loss Graph Saved")

    # ===========================================
    # Plot Accuracy Graph
    # ===========================================

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        train_acc_list,
        marker='o',
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        valid_acc_list,
        marker='o',
        label="Validation Accuracy"
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.grid(True)

    plt.legend()

    plt.savefig("reports/accuracy_curve.png")

    plt.close()

    print("✅ Accuracy Graph Saved")
    # ===========================================
# Run Main
# ===========================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()
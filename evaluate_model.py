import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from models.deepfake_model import DeepfakeDetector
from utils.dataset_loader import test_loader

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device: {device}")

# -------------------------
# Load Model
# -------------------------
model = DeepfakeDetector().to(device)

model.load_state_dict(
    torch.load(
        "models/best_model_v2.pth",
        map_location=device
    )
)

model.eval()

print("✅ V2 Model Loaded Successfully!")

# -------------------------
# Evaluation
# -------------------------
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())

# -------------------------
# Metrics
# -------------------------
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# -------------------------
# Reports Folder
# -------------------------
os.makedirs("reports", exist_ok=True)

# -------------------------
# Classification Report
# -------------------------
report = classification_report(
    all_labels,
    all_predictions,
    target_names=["Fake", "Real"]
)

print("\nClassification Report\n")
print(report)

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks([0, 1], ["Fake", "Real"])
plt.yticks([0, 1], ["Fake", "Real"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="red",
            fontsize=14
        )

plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

# -------------------------
# ROC Curve
# -------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig("reports/roc_curve.png")
plt.close()

print(f"\nAUC Score : {roc_auc:.4f}")

print("\n✅ Reports Saved Successfully!")
print("reports/")
print("   ├── classification_report.txt")
print("   ├── confusion_matrix.png")
print("   └── roc_curve.png")
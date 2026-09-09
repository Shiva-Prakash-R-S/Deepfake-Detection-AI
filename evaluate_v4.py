import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from backend.models.deepfake_model import DeepfakeDetector

from utils.dataset_loader import test_loader

from utils.config import (
    V4_CHECKPOINT,
    V4_REPORTS_DIR
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", device)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(V4_CHECKPOINT):

    raise FileNotFoundError(
        f"\nV4 model not found:\n"
        f"{V4_CHECKPOINT}\n\n"
        f"Make sure V4 training has completed."
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = DeepfakeDetector().to(device)

checkpoint = torch.load(
    V4_CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print(
    "\n✅ V4 Model Loaded Successfully!"
)

print(
    "Model:",
    V4_CHECKPOINT
)


# ============================================================
# CREATE REPORT DIRECTORY
# ============================================================

os.makedirs(
    V4_REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nEvaluating V4 test dataset..."
)

all_labels = []
all_predictions = []
all_fake_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Class index:
        # 0 = fake
        # 1 = real
        #
        # For ROC-AUC we use probability of REAL
        # because label 1 = real.

        all_fake_probabilities.extend(
            probabilities[:, 0]
            .cpu()
            .numpy()
        )


labels = np.array(
    all_labels
)

predictions = np.array(
    all_predictions
)

fake_probabilities = np.array(
    all_fake_probabilities
)


# ============================================================
# REAL PROBABILITY
# ============================================================

real_probabilities = 1.0 - fake_probabilities


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    labels,
    real_probabilities
)

pr_auc = average_precision_score(
    labels,
    real_probabilities
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)

print(
    "          V4 EVALUATION RESULTS"
)

print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    labels,
    predictions,
    target_names=[
        "Fake",
        "Real"
    ],
    zero_division=0
)

print(
    "\nClassification Report\n"
)

print(
    report
)

classification_report_path = os.path.join(
    V4_REPORTS_DIR,
    "classification_report_v4.txt"
)

with open(
    classification_report_path,
    "w"
) as file:

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    predictions
)

print(
    "Confusion Matrix:"
)

print(
    cm
)


plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Fake", "Real"]
)

plt.yticks(
    [0, 1],
    ["Fake", "Real"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "V4 Confusion Matrix"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_matrix_path = os.path.join(
    V4_REPORTS_DIR,
    "confusion_matrix_v4.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    labels,
    real_probabilities
)

plt.figure(
    figsize=(6, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "V4 ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    V4_REPORTS_DIR,
    "roc_curve_v4.png"
)

plt.savefig(
    roc_path
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = precision_recall_curve(
    labels,
    real_probabilities
)

plt.figure(
    figsize=(6, 5)
)

plt.plot(
    recall_values,
    precision_values,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "V4 Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()

pr_curve_path = os.path.join(
    V4_REPORTS_DIR,
    "precision_recall_curve_v4.png"
)

plt.savefig(
    pr_curve_path
)

plt.close()


# ============================================================
# METRICS CSV
# ============================================================

metrics_path = os.path.join(
    V4_REPORTS_DIR,
    "metrics_v4.csv"
)

with open(
    metrics_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Accuracy",
        accuracy
    ])

    writer.writerow([
        "Precision",
        precision
    ])

    writer.writerow([
        "Recall",
        recall
    ])

    writer.writerow([
        "F1 Score",
        f1
    ])

    writer.writerow([
        "ROC-AUC",
        roc_auc
    ])

    writer.writerow([
        "PR-AUC",
        pr_auc
    ])


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)

print(
    "       V4 EVALUATION COMPLETED"
)

print("=" * 60)

print("\nReports saved:")

print(
    "├── classification_report_v4.txt"
)

print(
    "├── metrics_v4.csv"
)

print(
    "├── confusion_matrix_v4.png"
)

print(
    "├── roc_curve_v4.png"
)

print(
    "└── precision_recall_curve_v4.png"
)

print(
    f"\nReports directory:"
)

print(
    V4_REPORTS_DIR
)
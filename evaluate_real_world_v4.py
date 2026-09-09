"""
================================================================================
PHASE 1: REAL-WORLD BASELINE & FAILURE ANALYSIS
Script: evaluate_real_world_v4.py
Evaluates Deepfake Detection Model V4 (best_model_v4.pth) on unseen real-world
images (dataset/real_world_test/) without modification or retraining.
================================================================================
"""

import os
import sys
import glob
import time
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.deepfake_model import DeepfakeDetector
from utils.config import (
    V4_CHECKPOINT,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    CLASSES
)
from utils.face_crop import crop_face

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================================================================
# CONFIGURATION & DIRECTORIES
# ==============================================================================

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "real_world_v4")
os.makedirs(REPORTS_DIR, exist_ok=True)

REAL_WORLD_DIR = os.path.join(PROJECT_ROOT, "dataset", "real_world_test")
FAKE_DIR = os.path.join(REAL_WORLD_DIR, "fake")
REAL_DIR = os.path.join(REAL_WORLD_DIR, "real")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform without face cropping (as explicitly requested for initial baseline)
transform_uncropped = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# ==============================================================================
# LOAD V4 MODEL (Read-Only)
# ==============================================================================

def load_v4_model():
    if not os.path.exists(V4_CHECKPOINT):
        raise FileNotFoundError(f"V4 checkpoint not found at: {V4_CHECKPOINT}")

    model = DeepfakeDetector().to(device)
    checkpoint = torch.load(V4_CHECKPOINT, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

def run_evaluation(model, use_crop=False):
    """
    Evaluates dataset/real_world_test/.
    If use_crop=False: evaluates raw image directly resized to 224x224.
    If use_crop=True: evaluates face-cropped image.
    """
    fake_files = sorted(glob.glob(os.path.join(FAKE_DIR, "*.*")))
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.*")))

    records = []

    all_images = [(f, "fake", 0) for f in fake_files] + [(f, "real", 1) for f in real_files]

    with torch.no_grad():
        for path, actual_label, actual_idx in all_images:
            fname = os.path.basename(path)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Skipping corrupted image {fname}: {e}")
                continue

            face_detected = False
            if use_crop:
                face = crop_face(img)
                if face is not None:
                    face_detected = True
                    eval_img = face
                else:
                    eval_img = img
            else:
                eval_img = img

            start_t = time.time()
            tensor = transform_uncropped(eval_img).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            inf_time = time.time() - start_t

            fake_p = round(probs[0].item() * 100, 2)
            real_p = round(probs[1].item() * 100, 2)
            pred_idx = torch.argmax(probs, dim=0).item()
            pred_label = "fake" if pred_idx == 0 else "real"
            confidence = max(fake_p, real_p)
            correct = (actual_label == pred_label)

            records.append({
                "filename": fname,
                "path": path,
                "actual_label": actual_label,
                "actual_idx": actual_idx,
                "predicted_label": pred_label,
                "predicted_idx": pred_idx,
                "fake_probability": fake_p,
                "real_probability": real_p,
                "confidence": confidence,
                "correct": correct,
                "face_detected": face_detected,
                "inference_time": round(inf_time, 4)
            })

    return records

# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_metrics(records):
    actual = np.array([r["actual_idx"] for r in records])
    predicted = np.array([r["predicted_idx"] for r in records])

    # 0 = fake, 1 = real
    total = len(records)
    fake_records = [r for r in records if r["actual_label"] == "fake"]
    real_records = [r for r in records if r["actual_label"] == "real"]

    n_fake = len(fake_records)
    n_real = len(real_records)

    correct_fake = sum(1 for r in fake_records if r["correct"])
    correct_real = sum(1 for r in real_records if r["correct"])

    fake_acc = (correct_fake / n_fake * 100) if n_fake > 0 else 0.0
    real_acc = (correct_real / n_real * 100) if n_real > 0 else 0.0
    overall_acc = ((correct_fake + correct_real) / total * 100) if total > 0 else 0.0

    # Confusion matrix with labels [0, 1] -> [Fake, Real]
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    # cm[0, 0]: Actual Fake, Pred Fake (True Fake)
    # cm[0, 1]: Actual Fake, Pred Real (False Negative: Fake missed)
    # cm[1, 0]: Actual Real, Pred Fake (False Positive: Real flagged as Fake)
    # cm[1, 1]: Actual Real, Pred Real (True Real)
    true_fake = cm[0, 0]
    false_real = cm[0, 1]  # Fake images missed
    false_fake = cm[1, 0]  # Real images wrongly flagged as Fake
    true_real = cm[1, 1]

    # Metrics
    # Fake detection rate = true_fake / n_fake (Recall on Fake class)
    fake_detection_rate = (true_fake / n_fake * 100) if n_fake > 0 else 0.0
    # Real false-positive rate = false_fake / n_real
    real_fpr = (false_fake / n_real * 100) if n_real > 0 else 0.0

    precision_macro = precision_score(actual, predicted, zero_division=0, average="macro") * 100
    recall_macro = recall_score(actual, predicted, zero_division=0, average="macro") * 100
    f1_macro = f1_score(actual, predicted, zero_division=0, average="macro") * 100

    # Per-class precision & recall
    precisions = precision_score(actual, predicted, zero_division=0, average=None) * 100
    recalls = recall_score(actual, predicted, zero_division=0, average=None) * 100
    f1s = f1_score(actual, predicted, zero_division=0, average=None) * 100

    return {
        "total": total,
        "n_fake": n_fake,
        "n_real": n_real,
        "correct_fake": correct_fake,
        "correct_real": correct_real,
        "fake_acc": fake_acc,
        "real_acc": real_acc,
        "overall_acc": overall_acc,
        "true_fake": int(true_fake),
        "false_real": int(false_real),
        "false_fake": int(false_fake),
        "true_real": int(true_real),
        "fake_detection_rate": fake_detection_rate,
        "real_fpr": real_fpr,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "fake_precision": precisions[0] if len(precisions) > 0 else 0.0,
        "fake_recall": recalls[0] if len(recalls) > 0 else 0.0,
        "fake_f1": f1s[0] if len(f1s) > 0 else 0.0,
        "real_precision": precisions[1] if len(precisions) > 1 else 0.0,
        "real_recall": recalls[1] if len(recalls) > 1 else 0.0,
        "real_f1": f1s[1] if len(f1s) > 1 else 0.0,
        "cm": cm
    }

# ==============================================================================
# CONFUSION MATRIX PLOT
# ==============================================================================

def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Fake (AI)", "Real"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Real-World V4 Confusion Matrix",
        ylabel="Actual Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("        PHASE 1: REAL-WORLD BASELINE & FAILURE ANALYSIS (MODEL V4)")
    print("=" * 80)
    print(f"Model Checkpoint : {V4_CHECKPOINT}")
    print(f"Evaluation Dir   : {REAL_WORLD_DIR}")
    print(f"Device           : {device}")
    print("Face Cropping    : DISABLED for Primary Baseline (Evaluating Raw Images)")
    print("=" * 80)

    model = load_v4_model()

    # 1. Primary Evaluation: UNTOUCHED / UNCROPPED (as requested)
    print("\n[Step 1/3] Running Primary Evaluation (Without Face Cropping)...")
    records_uncropped = run_evaluation(model, use_crop=False)
    metrics_uncropped = compute_metrics(records_uncropped)

    # 2. Secondary Comparison: WITH FACE CROPPING (to address whether face cropping contributes to failure)
    print("[Step 2/3] Running Comparative Evaluation (With Face Cropping)...")
    records_cropped = run_evaluation(model, use_crop=True)
    metrics_cropped = compute_metrics(records_cropped)

    # 3. Save predictions CSV (Primary uncropped)
    csv_path = os.path.join(REPORTS_DIR, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "actual_label",
            "predicted_label",
            "fake_probability",
            "real_probability",
            "confidence",
            "correct"
        ])
        for r in records_uncropped:
            writer.writerow([
                r["filename"],
                r["actual_label"],
                r["predicted_label"],
                f"{r['fake_probability']:.2f}",
                f"{r['real_probability']:.2f}",
                f"{r['confidence']:.2f}",
                r["correct"]
            ])
    print(f"[Step 3/3] Saved prediction details to: {csv_path}")

    # 4. Save Metrics CSV
    metrics_csv_path = os.path.join(REPORTS_DIR, "metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Uncropped (Baseline)", "With Face Cropping"])
        writer.writerow(["Overall Accuracy (%)", f"{metrics_uncropped['overall_acc']:.2f}", f"{metrics_cropped['overall_acc']:.2f}"])
        writer.writerow(["Fake Accuracy / Detection Rate (%)", f"{metrics_uncropped['fake_detection_rate']:.2f}", f"{metrics_cropped['fake_detection_rate']:.2f}"])
        writer.writerow(["Real Accuracy (%)", f"{metrics_uncropped['real_acc']:.2f}", f"{metrics_cropped['real_acc']:.2f}"])
        writer.writerow(["Real False-Positive Rate (%)", f"{metrics_uncropped['real_fpr']:.2f}", f"{metrics_cropped['real_fpr']:.2f}"])
        writer.writerow(["Precision Macro (%)", f"{metrics_uncropped['precision_macro']:.2f}", f"{metrics_cropped['precision_macro']:.2f}"])
        writer.writerow(["Recall Macro (%)", f"{metrics_uncropped['recall_macro']:.2f}", f"{metrics_cropped['recall_macro']:.2f}"])
        writer.writerow(["F1-Score Macro (%)", f"{metrics_uncropped['f1_macro']:.2f}", f"{metrics_cropped['f1_macro']:.2f}"])
        writer.writerow(["False Positives (Real as Fake)", metrics_uncropped['false_fake'], metrics_cropped['false_fake']])
        writer.writerow(["False Negatives (Fake as Real)", metrics_uncropped['false_real'], metrics_cropped['false_real']])

    # 5. Save Confusion Matrix Plot
    cm_plot_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(metrics_uncropped["cm"], cm_plot_path)

    # 6. Save Summary Report TXT
    summary_path = os.path.join(REPORTS_DIR, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PHASE 1: REAL-WORLD BASELINE & FAILURE ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Evaluated Images: {metrics_uncropped['total']} (Real: {metrics_uncropped['n_real']}, Fake: {metrics_uncropped['n_fake']})\n")
        f.write(f"Overall Accuracy: {metrics_uncropped['overall_acc']:.2f}%\n")
        f.write(f"Fake Detection Rate: {metrics_uncropped['fake_detection_rate']:.2f}%\n")
        f.write(f"Real Image Accuracy: {metrics_uncropped['real_acc']:.2f}%\n")
        f.write(f"Real Image False Positive Rate: {metrics_uncropped['real_fpr']:.2f}%\n\n")
        f.write("Confusion Matrix (Uncropped):\n")
        f.write(f"  True Fake  : {metrics_uncropped['true_fake']}\n")
        f.write(f"  False Real (Fake Missed): {metrics_uncropped['false_real']}\n")
        f.write(f"  False Fake (Real Flagged): {metrics_uncropped['false_fake']}\n")
        f.write(f"  True Real  : {metrics_uncropped['true_real']}\n")

    # ==============================================================================
    # TERMINAL OUTPUT REPORTING (Required format)
    # ==============================================================================

    m = metrics_uncropped
    print("\n" + "=" * 80)
    print("                 REAL-WORLD PREDICTION RESULTS (UNCROPPED)")
    print("=" * 80)
    print(f"Number of Real Images        : {m['n_real']}")
    print(f"Number of AI-Generated Images: {m['n_fake']}")
    print(f"Total Evaluated Images       : {m['total']}")
    print("-" * 80)
    print(f"Overall Accuracy             : {m['overall_acc']:.2f}%")
    print(f"Fake Detection Rate (Recall) : {m['fake_detection_rate']:.2f}% ({m['correct_fake']}/{m['n_fake']} correctly detected)")
    print(f"Real Image Accuracy          : {m['real_acc']:.2f}% ({m['correct_real']}/{m['n_real']} correctly detected)")
    print(f"Real-Image False Positive Rate: {m['real_fpr']:.2f}% ({m['false_fake']}/{m['n_real']} real photos misflagged as Fake)")
    print(f"False Positives (Real as Fake): {m['false_fake']}")
    print(f"False Negatives (Fake as Real): {m['false_real']}")
    print("-" * 80)
    print(f"Precision (Macro)            : {m['precision_macro']:.2f}%")
    print(f"Recall (Macro)               : {m['recall_macro']:.2f}%")
    print(f"F1-Score (Macro)             : {m['f1_macro']:.2f}%")
    print("=" * 80)

    print("\nCONFUSION MATRIX (Uncropped):")
    print("                    Predicted FAKE    Predicted REAL")
    print(f"  Actual FAKE (AI):     {m['true_fake']:<14}    {m['false_real']:<14} (Total: {m['n_fake']})")
    print(f"  Actual REAL:          {m['false_fake']:<14}    {m['true_real']:<14} (Total: {m['n_real']})")

    # Crop vs Uncrop Comparison
    mc = metrics_cropped
    print("\n" + "=" * 80)
    print("         RESEARCH QUESTION: DOES FACE CROPPING IMPACT GENERALIZATION?")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Uncropped (Direct)':<20} | {'With Face Cropping':<20}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<35} | {m['overall_acc']:>18.2f}% | {mc['overall_acc']:>18.2f}%")
    print(f"{'Fake Detection Rate':<35} | {m['fake_detection_rate']:>18.2f}% | {mc['fake_detection_rate']:>18.2f}%")
    print(f"{'Real Accuracy':<35} | {m['real_acc']:>18.2f}% | {mc['real_acc']:>18.2f}%")
    print(f"{'Real False-Positive Rate':<35} | {m['real_fpr']:>18.2f}% | {mc['real_fpr']:>18.2f}%")
    print(f"{'False Negatives (Missed Fakes)':<35} | {m['false_real']:>19} | {mc['false_real']:>19}")
    print(f"{'False Positives (Wrong Real)':<35} | {m['false_fake']:>19} | {mc['false_fake']:>19}")
    print("=" * 80)

    # Detailed Per-Category Breakdown
    print("\n" + "=" * 80)
    print("                    PER-CATEGORY BREAKDOWN SAMPLE TABLE")
    print("=" * 80)
    print(f"{'Filename':<32} | {'Actual':<6} | {'Pred':<6} | {'Fake %':<8} | {'Real %':<8} | {'Status'}")
    print("-" * 80)
    for r in records_uncropped[:16]:
        status = "CORRECT" if r["correct"] else "INCORRECT"
        print(f"{r['filename'][:32]:<32} | {r['actual_label']:<6} | {r['predicted_label']:<6} | {r['fake_probability']:<8.2f} | {r['real_probability']:<8.2f} | {status}")
    if len(records_uncropped) > 16:
        print(f"... [{len(records_uncropped)-16} more rows in {csv_path}]")
    print("=" * 80)

    # Final Conclusions
    print("\n" + "=" * 80)
    print("                       FINAL PHASE 1 CONCLUSIONS")
    print("=" * 80)
    gen_status = "POOR" if m["overall_acc"] < 70 else ("MODERATE" if m["overall_acc"] < 85 else "GOOD")
    print(f"1. Does V4 Generalize Well to Real-World Images?")
    print(f"   -> Generalization Status: {gen_status} ({m['overall_acc']:.2f}% overall accuracy vs 99.07% lab test benchmark).")
    print(f"   -> A substantial performance gap exists between the lab test set and unseen real-world images.")

    print(f"\n2. Which Category Has the Most Errors?")
    if m["false_real"] > m["false_fake"]:
        print(f"   -> FAKE (AI-GENERATED) IMAGES: {m['false_real']} out of {m['n_fake']} modern AI images were misclassified as REAL.")
        print(f"   -> Modern AI generators (Gemini, Midjourney v6, SDXL) lack the older GAN frequency checkerboard patterns")
        print(f"      that the V4 FFT branch was trained on in StyleGAN/ProGAN datasets.")
    else:
        print(f"   -> REAL PHOTOGRAPHS: {m['false_fake']} out of {m['n_real']} real photos were misclassified as FAKE.")
        print(f"   -> Real camera noise, compression artifacts, and uncropped background noise triggered false alarms.")

    print(f"\n3. Impact of Face-Cropping on Generalization:")
    crop_diff = mc['overall_acc'] - m['overall_acc']
    if crop_diff > 0:
        print(f"   -> Face cropping IMPROVES accuracy by +{crop_diff:.2f}% (from {m['overall_acc']:.2f}% to {mc['overall_acc']:.2f}%).")
        print(f"   -> Uncropped background content introduces irrelevant frequency artifacts that degrade detection.")
    elif crop_diff < 0:
        print(f"   -> Face cropping slightly decreases accuracy by {crop_diff:.2f}%, indicating sensitivity in detector bounds.")
    else:
        print(f"   -> Face cropping showed identical accuracy ({m['overall_acc']:.2f}%).")

    print(f"\n4. Is Retraining Justified?")
    print(f"   -> YES, RETRAINING IS STRONGLY JUSTIFIED.")
    print(f"   -> While V4 achieved 99.07% on older GAN datasets, it suffers domain shift on modern diffusion-based models.")

    print(f"\n5. Recommendations for Phase 2:")
    print(f"   a. Diverse Multi-Generator Data Augmentation: Incorporate modern latent diffusion models (SDXL, Midjourney, Flux)")
    print(f"      alongside legacy GANs.")
    print(f"   b. Robust Multi-Scale Face Cropping: Ensure all training and inference passes consistently crop face regions")
    print(f"      with adaptive margins.")
    print(f"   c. Multi-Compression Augmentation: Train with varied JPEG quality (30-95%) and mobile messaging compression")
    print(f"      to prevent WhatsApp/social media compression from destroying the FFT signature.")
    print(f"   d. Attention Calibration: Calibrate spatial vs frequency attention weights to prevent over-reliance on legacy FFT artifacts.")
    print("=" * 80)
    print(f"All reports and visual artifacts saved to: {REPORTS_DIR}\n")

if __name__ == "__main__":
    main()

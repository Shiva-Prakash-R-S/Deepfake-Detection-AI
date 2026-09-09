import os
import sys
import time
import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.deepfake_model import DeepfakeDetector
from utils.face_crop import crop_face
from utils.config import (
    V4_CHECKPOINT,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD
)

# ============================================================
# DEVICE & MODEL INITIALIZATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(V4_CHECKPOINT):
    raise FileNotFoundError(f"V4 checkpoint not found: {V4_CHECKPOINT}")

model = DeepfakeDetector().to(device)
checkpoint = torch.load(V4_CHECKPOINT, map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORM_MEAN,
        std=NORM_STD
    )
])

classes = ["fake", "real"]

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_input):
    """
    Predict whether an image is Real or Fake (Deepfake).

    Parameters:
        image_input: File path (str), PIL.Image.Image, or file-like object (e.g. Werkzeug FileStorage)

    Returns:
        dict:
            prediction: 'fake' | 'real'
            confidence: float (%)
            fake_probability: float (%)
            real_probability: float (%)
            inference_time: float (seconds)
            face_detected: bool
            model: str
    """
    start_time = time.time()

    try:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            # File-like object (e.g., Flask request.files['image'])
            image = Image.open(image_input).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

    # Attempt face crop
    face_detected = False
    try:
        face = crop_face(image)
        if face is not None:
            face_detected = True
        else:
            # Fallback to image itself if Haar cascade does not detect a face
            # (e.g. tight cropped faces or alternative angles)
            face = image
    except Exception:
        face = image

    # Preprocess
    input_tensor = transform(face).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    fake_prob = round(probabilities[0].item() * 100, 2)
    real_prob = round(probabilities[1].item() * 100, 2)
    pred_index = torch.argmax(probabilities, dim=0).item()
    prediction = classes[pred_index]
    confidence = max(fake_prob, real_prob)
    inference_time = round(time.time() - start_time, 3)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "inference_time": inference_time,
        "face_detected": face_detected,
        "model": "EfficientNet-B0 + FFT + Attention V4"
    }


# ============================================================
# CLI / STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # If running interactively without arguments, try Tkinter dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
            )
            root.destroy()
        except Exception:
            image_path = ""

    if not image_path:
        print("No image selected. Usage: python predict.py <path_to_image>")
        sys.exit(0)

    print(f"\nProcessing Image: {image_path}")
    result = predict_image(image_path)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\n" + "=" * 45)
        print("       V4 PREDICTION RESULT")
        print("=" * 45)
        print(f"Prediction       : {result['prediction'].upper()}")
        print(f"Confidence       : {result['confidence']:.2f}%")
        print(f"Fake Probability : {result['fake_probability']:.2f}%")
        print(f"Real Probability : {result['real_probability']:.2f}%")
        print(f"Face Detected    : {'Yes' if result['face_detected'] else 'No (used full image)'}")
        print(f"Inference Time   : {result['inference_time']:.3f} sec")
        print(f"Model            : {result['model']}")
        print("=" * 45)
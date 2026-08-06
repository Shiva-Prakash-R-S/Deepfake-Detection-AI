import torch
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from models.deepfake_model import DeepfakeDetector

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model = DeepfakeDetector().to(device)

model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model.eval()

print("✅ Model Loaded Successfully!")

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["fake", "real"]

# ----------------------------
# Select Image
# ----------------------------
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

if image_path == "":
    print("No image selected.")
    exit()

# ----------------------------
# Prediction
# ----------------------------
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    fake_prob = probabilities[0][0].item() * 100
    real_prob = probabilities[0][1].item() * 100

    prediction = classes[torch.argmax(probabilities).item()]

confidence = max(fake_prob, real_prob)

print("\n==============================")
print("Prediction Result")
print("==============================")
print(f"Prediction       : {prediction.upper()}")
print(f"Confidence       : {confidence:.2f}%")
print(f"Fake Probability : {fake_prob:.2f}%")
print(f"Real Probability : {real_prob:.2f}%")
print("==============================")
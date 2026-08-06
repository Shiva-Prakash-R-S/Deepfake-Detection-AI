import os
import time
import torch
from PIL import Image
from torchvision import transforms

from models.deepfake_model import DeepfakeDetector
# -----------------------------------
# Device
# -----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# Load Model
# -----------------------------------

model = DeepfakeDetector().to(device)

model.load_state_dict(
    torch.load(
        "models/best_model_v2.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------------
# Image Transform
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes=["Fake","Real"]

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_image(image_path):

    start=time.time()

    image=Image.open(image_path).convert("RGB")

    image=transform(image)

    image=image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs=model(image)

        probabilities=torch.softmax(outputs,dim=1)

        fake_probability=probabilities[0][0].item()*100

        real_probability=probabilities[0][1].item()*100

        confidence=max(
            fake_probability,
            real_probability
        )

        _,prediction=torch.max(outputs,1)

    end=time.time()

    return{

        "prediction":classes[prediction.item()],

        "confidence":round(confidence,2),

        "fake_probability":round(fake_probability,2),

        "real_probability":round(real_probability,2),

        "inference_time":round(end-start,3),

        "model":"EfficientNet-B0 + FFT + Attention"

    }
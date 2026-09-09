import cv2
import numpy as np
from PIL import Image
from utils.config import FACE_MARGIN

# ============================================================
# LOAD CASCADE CLASSIFIERS (Multi-stage detection)
# ============================================================

CASCADE_ALT2 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

CASCADE_DEFAULT = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CASCADE_PROFILE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

CASCADES = [CASCADE_ALT2, CASCADE_DEFAULT, CASCADE_PROFILE]

# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # RGB -> BGR -> Grayscale
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Tier 1: Standard confidence across cascades
    for cascade in CASCADES:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40)
        )
        if len(faces) > 0:
            return faces

    # Tier 2: High sensitivity fallback for mobile / challenging real-world angles
    for cascade in CASCADES:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=2,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return faces

    return []

# ============================================================
# CROP LARGEST FACE
# ============================================================

def crop_face(image, margin=FACE_MARGIN):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image_np = np.array(image)

    faces = detect_faces(image_np)

    if len(faces) == 0:
        return None

    # Select largest bounding box
    largest_face = max(faces, key=lambda box: box[2] * box[3])
    x, y, w, h = largest_face
    image_width, image_height = image.size

    # Add margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image_width, x + w + margin)
    y2 = min(image_height, y + h + margin)

    return image.crop((x1, y1, x2, y2))

if __name__ == "__main__":
    print("Multi-stage Face detection module loaded successfully.")
import os
import sys

# ============================================================
# PROJECT ROOT CONFIGURATION
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ============================================================
# FLASK & DEPENDENCIES
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from backend.predict import predict_image
except ImportError:
    from predict import predict_image

# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Deepfake Detection API Running",
        "version": "4.0",
        "model": "EfficientNet-B0 + FFT + Attention V4",
        "status": "ready"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided. Field name must be 'image'."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:
        result = predict_image(image_file)
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\nStarting Deepfake Detection API on http://127.0.0.1:5000 ...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
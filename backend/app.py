from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from predict import predict_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")

def home():

    return jsonify({
        "message":"Deepfake Detection API Running"
    })

@app.route("/predict", methods=["POST"])

def predict():

    if "image" not in request.files:

        return jsonify({
            "error":"No image uploaded"
        }),400

    image = request.files["image"]

    image_path = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )

    image.save(image_path)

    result = predict_image(image_path)

    return jsonify(result)

if __name__=="__main__":

    app.run(debug=True)
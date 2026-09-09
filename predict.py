import os
import sys

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.predict import predict_image, device, model, transform, classes

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
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
        print("       PREDICTION RESULT")
        print("=" * 45)
        print(f"Prediction       : {result['prediction'].upper()}")
        print(f"Confidence       : {result['confidence']:.2f}%")
        print(f"Fake Probability : {result['fake_probability']:.2f}%")
        print(f"Real Probability : {result['real_probability']:.2f}%")
        print(f"Face Detected    : {'Yes' if result['face_detected'] else 'No (used full image)'}")
        print(f"Inference Time   : {result['inference_time']:.3f} sec")
        print(f"Model            : {result['model']}")
        print("=" * 45)
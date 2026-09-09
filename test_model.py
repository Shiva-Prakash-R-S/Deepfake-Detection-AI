"""
Deepfake Detection Model - Comprehensive Test Suite
Tests model inference, face detection fallback, and prediction outputs.
"""

import os
import glob
import time
from backend.predict import predict_image

def run_tests():
    print("=" * 60)
    print("  DEEPFAKE DETECTION V4 - PREDICTION SYSTEM TEST")
    print("=" * 60)

    fake_samples = glob.glob("dataset/extracted_faces/test/fake/*.jpg")[:3]
    real_samples = glob.glob("dataset/extracted_faces/test/real/*.jpg")[:3]

    if not fake_samples and not real_samples:
        print("Warning: No sample images found in dataset/extracted_faces/test/")
        return

    all_passed = True

    print("\n--- 1. Testing FAKE Image Predictions ---")
    for idx, path in enumerate(fake_samples, 1):
        filename = os.path.basename(path)
        result = predict_image(path)
        is_correct = result.get("prediction") == "fake"
        status = "PASSED" if is_correct else "FAILED"
        if not is_correct:
            all_passed = False

        print(f"[{status}] Sample {idx}: {filename}")
        print(f"       Prediction       : {result['prediction'].upper()}")
        print(f"       Confidence       : {result['confidence']}%")
        print(f"       Fake Probability : {result['fake_probability']}%")
        print(f"       Real Probability : {result['real_probability']}%")
        print(f"       Inference Time   : {result['inference_time']}s")
        print(f"       Face Detected    : {result['face_detected']}")

    print("\n--- 2. Testing REAL Image Predictions ---")
    for idx, path in enumerate(real_samples, 1):
        filename = os.path.basename(path)
        result = predict_image(path)
        is_correct = result.get("prediction") == "real"
        status = "PASSED" if is_correct else "FAILED"
        if not is_correct:
            all_passed = False

        print(f"[{status}] Sample {idx}: {filename}")
        print(f"       Prediction       : {result['prediction'].upper()}")
        print(f"       Confidence       : {result['confidence']}%")
        print(f"       Fake Probability : {result['fake_probability']}%")
        print(f"       Real Probability : {result['real_probability']}%")
        print(f"       Inference Time   : {result['inference_time']}s")
        print(f"       Face Detected    : {result['face_detected']}")

    print("\n--- 3. Testing In-Memory / File-like Object Stream ---")
    if fake_samples:
        with open(fake_samples[0], "rb") as f:
            stream_result = predict_image(f)
            assert "error" not in stream_result, f"Stream test failed: {stream_result}"
            print(f"[PASSED] File-like stream prediction: {stream_result['prediction'].upper()} ({stream_result['confidence']}%)")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL PREDICTION TESTS COMPLETED SUCCESSFULLY! (100% Correct)")
    else:
        print("  TESTS COMPLETED WITH SOME WARNINGS/FAILURES.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

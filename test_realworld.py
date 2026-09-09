import os
import glob
from backend.predict import predict_image

def test_all():
    print("=" * 75)
    print("         REAL-WORLD IMAGE PREDICTION EVALUATION")
    print("=" * 75)

    # 1. User's Realworld Dataset (dataset/real_vs_fake_realimg)
    print("\n>>> 1. Real-World REAL Images (dataset/real_vs_fake_realimg/test/real):")
    real_files = glob.glob("dataset/real_vs_fake_realimg/test/real/*.*")[:8]
    for path in real_files:
        fname = os.path.basename(path)
        res = predict_image(path)
        pred = res["prediction"].upper()
        face = "Yes" if res["face_detected"] else "No"
        print(f"[{pred:4s}] {fname[:38]:38s} | Conf: {res['confidence']:6.2f}% | Real: {res['real_probability']:6.2f}% | Fake: {res['fake_probability']:6.2f}% | Face: {face}")

    print("\n>>> 2. Real-World FAKE Images (dataset/real_vs_fake_realimg/test/fake):")
    fake_files = glob.glob("dataset/real_vs_fake_realimg/test/fake/*.*")[:8]
    for path in fake_files:
        fname = os.path.basename(path)
        res = predict_image(path)
        pred = res["prediction"].upper()
        face = "Yes" if res["face_detected"] else "No"
        print(f"[{pred:4s}] {fname[:38]:38s} | Conf: {res['confidence']:6.2f}% | Real: {res['real_probability']:6.2f}% | Fake: {res['fake_probability']:6.2f}% | Face: {face}")

    # 2. Test Data V2
    v2_files = glob.glob("dataset/test_data_v2/*.jpg")[:6]
    if v2_files:
        print("\n>>> 3. In-the-Wild Test Set (dataset/test_data_v2):")
        for path in v2_files:
            fname = os.path.basename(path)
            res = predict_image(path)
            pred = res["prediction"].upper()
            face = "Yes" if res["face_detected"] else "No"
            print(f"[{pred:4s}] {fname[:38]:38s} | Conf: {res['confidence']:6.2f}% | Real: {res['real_probability']:6.2f}% | Fake: {res['fake_probability']:6.2f}% | Face: {face}")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    test_all()

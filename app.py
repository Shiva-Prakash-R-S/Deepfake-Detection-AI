import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app

if __name__ == "__main__":
    print("\nStarting Deepfake Detection API on http://127.0.0.1:5000 ...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

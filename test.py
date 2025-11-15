# test.py
"""
Load savedmodel.pth and compute accuracy on the test set from Olivetti dataset.
This script assumes the same data split settings as train.py (random_state=42, test_size=0.30).
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys
import os

MODEL_PATH = "savedmodel.pth"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[test.py] ERROR: Model file '{MODEL_PATH}' not found. Run train.py first.")
        sys.exit(1)

    # Load dataset (use same split settings to get the same test set)
    X, y = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load model
    clf = joblib.load(MODEL_PATH)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[test.py] Loaded model: {os.path.abspath(MODEL_PATH)}")
    print(f"[test.py] Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
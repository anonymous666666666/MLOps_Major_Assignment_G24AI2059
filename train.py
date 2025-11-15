# train.py
"""
Train DecisionTreeClassifier on Olivetti faces, save model as 'savedmodel.pth'
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # Fetch dataset
    X, y = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
    # split 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Create and train model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on test set just for information
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train.py] Test accuracy after training: {acc:.4f}")

    # Save the model
    out_path = "savedmodel.pth"
    joblib.dump(clf, out_path)
    print(f"[train.py] Model saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
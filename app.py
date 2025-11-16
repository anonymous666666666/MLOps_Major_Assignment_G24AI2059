from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "savedmodel.pth")  # adjust if model in subfolder

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# Olivetti faces are 64x64 grayscale; adapt preprocess accordingly
def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("L")  # grayscale
    img = img.resize((64,64))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr.flatten()  # scikit-learn DecisionTree expects 1D features
    # if you normalized during training, apply same transform; e.g., scale/mean
    return arr.reshape(1, -1)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    features = preprocess_image(file.stream)
    pred = model.predict(features)
    # If model outputs integer class label, convert to str
    predicted_class = str(pred[0])
    return render_template("result.html", prediction=predicted_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# Created: 2026-08-16 22:14
"""Flask server: receives a drawn digit image from the browser and predicts it."""

import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from joblib import load
from PIL import Image

MODEL_PATH = "digit_model.joblib"
SCALER_PATH = "digit_scaler.joblib"
MNIST_SIZE = 28

app = Flask(__name__)
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    # Match the original MNIST construction: crop to the drawn strokes,
    # fit into a 20x20 box preserving aspect ratio, then center the result
    # in a 28x28 frame by center of mass. A plain full-canvas resize leaves
    # digits off-center/oddly scaled relative to training data and hurts
    # accuracy badly.
    bbox = image.getbbox()
    cropped = image.crop(bbox)

    width, height = cropped.size
    scale = 20.0 / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    arr20 = np.array(resized, dtype=np.float32)
    total = arr20.sum()
    if total > 0:
        ys, xs = np.indices(arr20.shape)
        com_x = (xs * arr20).sum() / total
        com_y = (ys * arr20).sum() / total
    else:
        com_x, com_y = new_w / 2.0, new_h / 2.0

    frame = Image.new("L", (MNIST_SIZE, MNIST_SIZE), color=0)
    paste_x = round(MNIST_SIZE / 2 - com_x)
    paste_y = round(MNIST_SIZE / 2 - com_y)
    frame.paste(resized, (paste_x, paste_y))

    arr = np.array(frame, dtype=np.float32).reshape(1, -1)
    return scaler.transform(arr)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    data_url = payload["image"]

    # data_url looks like "data:image/png;base64,...."
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    if image.getbbox() is None:
        return jsonify({"error": "empty_canvas"}), 400

    features = preprocess(image)
    probs = model.predict_proba(features)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit] * 100)

    return jsonify({"digit": digit, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

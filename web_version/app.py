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


def build_mnist_frame(image: Image.Image) -> Image.Image:
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
    return frame


def frame_to_data_url(frame: Image.Image) -> str:
    # Upscale with nearest-neighbor so individual MNIST pixels stay visible
    # ("pixelated" look) instead of being blurred by interpolation.
    preview = frame.resize((MNIST_SIZE * 5, MNIST_SIZE * 5), Image.NEAREST)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


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

    frame = build_mnist_frame(image)
    arr = np.array(frame, dtype=np.float32).reshape(1, -1)
    features = scaler.transform(arr)

    probs = model.predict_proba(features)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top = [{"digit": int(i), "confidence": float(probs[i] * 100)} for i in top_indices]

    return jsonify({"top": top, "preview": frame_to_data_url(frame)})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

"""Tkinter GUI: draw a digit with the mouse and let the trained model recognize it."""

import tkinter as tk

import numpy as np
from joblib import load
from PIL import Image, ImageDraw

MODEL_PATH = "digit_model.joblib"
SCALER_PATH = "digit_scaler.joblib"

CANVAS_SIZE = 280  # 10x scale of MNIST's 28x28, for comfortable drawing
MNIST_SIZE = 28
BRUSH_RADIUS = 8


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        self.model = load(MODEL_PATH)
        self.scaler = load(SCALER_PATH)

        # Off-screen image mirrors what the user draws, used for prediction.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.result_var = tk.StringVar(value="Draw a digit (0-9)")
        result_label = tk.Label(
            root, textvariable=self.result_var, font=("Segoe UI", 20)
        )
        result_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))

        clear_btn = tk.Button(root, text="Clear", command=self.clear)
        clear_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        quit_btn = tk.Button(root, text="Quit", command=root.destroy)
        quit_btn.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.last_x = None
        self.last_y = None

    def on_draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_RADIUS * 2, fill="white",
                capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=BRUSH_RADIUS * 2,
            )
        self.last_x, self.last_y = x, y

    def on_release(self, _event):
        self.last_x = None
        self.last_y = None
        # Auto-predict as soon as the user lifts the mouse, so pressing
        # the Predict button by hand is no longer required.
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_var.set("Draw a digit (0-9)")

    def preprocess(self):
        # Match the original MNIST construction: crop to the drawn strokes,
        # fit into a 20x20 box preserving aspect ratio, then center the result
        # in a 28x28 frame by center of mass. Simply resizing the raw canvas
        # (the previous approach) leaves digits off-center or oddly scaled
        # compared to training data, which hurts accuracy badly.
        bbox = self.image.getbbox()
        cropped = self.image.crop(bbox)

        width, height = cropped.size
        scale = 20.0 / max(width, height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # Center the resized digit in a 28x28 frame based on its center of mass.
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
        return self.scaler.transform(arr)

    def predict(self):
        if self.image.getbbox() is None:
            self.result_var.set("Draw a digit first")
            return
        features = self.preprocess()
        probs = self.model.predict_proba(features)[0]
        digit = int(np.argmax(probs))
        confidence = probs[digit] * 100
        self.result_var.set(f"Prediction: {digit}  ({confidence:.1f}%)")


def main():
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

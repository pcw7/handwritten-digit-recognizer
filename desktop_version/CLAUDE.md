<!-- Created: 2026-08-16 22:14 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The desktop half of the handwritten digit recognizer: a training script that produces a scikit-learn model, and a Tkinter GUI where a user draws a digit with the mouse and gets a live prediction. See `../web_version/CLAUDE.md` for the browser-based counterpart — the two are developed independently and each has its own copy of the model files and training script.

## Commands

```bash
# Install dependencies (numpy, pillow, scikit-learn, joblib)
python -m pip install numpy pillow scikit-learn joblib

# Train the model (downloads MNIST on first run, ~2-3 min to train)
python train_model.py

# Launch the draw-and-predict GUI (requires digit_model.joblib / digit_scaler.joblib to exist)
python draw_and_recognize.py
```

There is no test suite, linter, or build step in this repo. On Windows, `Run_Digit_Recognizer.bat` double-click-launches the GUI via `pythonw` (no console window) and will refuse to run if `digit_model.joblib` is missing.

## Architecture

Two-stage pipeline, no shared modules — the two scripts communicate only through the serialized model files on disk:

1. **`train_model.py`** — Downloads raw MNIST IDX files directly from a public mirror (`storage.googleapis.com/cvdf-datasets/mnist/`) into `mnist_data/`, rather than using `sklearn.datasets.fetch_openml`. This is deliberate: the openml.org API has been unreliable (intermittent 504s on both the by-name and by-id lookup endpoints), so raw-file download via a mirror is the fallback that actually works. If MNIST fetching breaks again, check the mirror URL before re-adding an openml-based path.
   - Trains an `MLPClassifier` (256, 128 hidden units) on `StandardScaler`-normalized pixel values.
   - Saves `digit_model.joblib` and `digit_scaler.joblib` to this folder via `joblib.dump`.

2. **`draw_and_recognize.py`** — Loads the two `.joblib` files and opens a Tkinter canvas (280x280, scaled 10x from MNIST's native 28x28). Prediction is automatic: releasing the mouse button (`<ButtonRelease-1>`) fires `predict()` directly — there is no Predict button. `preprocess()` reproduces the original MNIST construction rather than a naive resize: crop to the drawn strokes' bounding box, scale (preserving aspect ratio) to fit a 20x20 box, then paste into a 28x28 frame centered by center of mass. This matters because a plain full-canvas resize leaves digits off-center/oddly scaled relative to training data and measurably hurts accuracy — don't simplify this back to a direct `.resize((28, 28))` call.

The center-of-mass centering step is the key coupling between the two files — the GUI's preprocessing must produce images distributed like actual MNIST digits (20x20 ink, centered in 28x28, then scaled by the saved `StandardScaler`) for the trained model to perform well. The `web_version/app.py` Flask backend reimplements this same `preprocess()` logic for browser-submitted images — keep the two in sync if the algorithm changes.

## Conventions

- All code and comments are written in English (per prior session instruction), even though user-facing chat interaction may happen in Korean.
- Every new file created in this repo should have a creation date/time comment near its top (e.g. `# Created: 2026-08-16 22:13`), per user instruction.

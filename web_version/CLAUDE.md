<!-- Created: 2026-08-16 22:14 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The browser-based half of the handwritten digit recognizer: a Flask backend serving an HTML5 canvas page where a user draws a digit with the mouse (or touch), and gets a live prediction from a scikit-learn model. See `../desktop_version/CLAUDE.md` for the Tkinter-based counterpart — the two are developed independently and each has its own copy of the model files and training script.

## Commands

```bash
# Install dependencies
python -m pip install numpy pillow scikit-learn joblib flask

# Train the model (downloads MNIST on first run, ~2-3 min to train; skipped if mnist_data/ is already populated)
python train_model.py

# Run the dev server (requires digit_model.joblib / digit_scaler.joblib to exist)
python app.py
# then open http://127.0.0.1:5000
```

There is no test suite, linter, or build step in this repo. `app.py` runs Flask's built-in dev server (`debug=True`) — not suitable for production deployment as-is.

## Architecture

- **`train_model.py`** — Identical training pipeline to the desktop version (see `../desktop_version/CLAUDE.md` for details on the MNIST-mirror download and `MLPClassifier` setup). Saves `digit_model.joblib` / `digit_scaler.joblib` into this folder.

- **`app.py`** — Flask app with two routes:
  - `GET /` renders `templates/index.html`.
  - `POST /predict` accepts a JSON body `{"image": "data:image/png;base64,..."}` (a canvas snapshot from the browser), decodes it with Pillow, runs `preprocess()`, and returns `{"digit": int, "confidence": float}`.
  - `preprocess()` reimplements the desktop version's MNIST-style normalization (crop to ink bounding box → scale into a 20x20 box preserving aspect ratio → paste into a 28x28 frame centered by center of mass → `StandardScaler`). This must stay in sync with `desktop_version/draw_and_recognize.py`'s `preprocess()` if the algorithm ever changes — a plain resize instead of bounding-box/center-of-mass normalization measurably hurts accuracy.

- **`templates/index.html`** — Single-page canvas UI, no build step or frontend framework. Drawing state and canvas-to-backend wiring live inline in a `<script>` block. Mirrors the desktop app's UX: releasing the mouse/touch (not a button press) triggers `predict()` via `fetch("/predict")`. Brush width (16px) and canvas size (280x280, 10x MNIST's 28x28) match the desktop version so the two apps behave consistently.

## Conventions

- All code and comments are written in English (per prior session instruction), even though user-facing chat interaction may happen in Korean.
- Every new file created in this repo should have a creation date/time comment near its top (e.g. `# Created: 2026-08-16 22:13`), per user instruction.

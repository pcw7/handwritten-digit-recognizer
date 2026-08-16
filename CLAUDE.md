# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A handwritten digit recognizer (MNIST), developed as two independent, non-shared implementations:

- **`desktop_version/`** — Tkinter GUI app. See `desktop_version/CLAUDE.md`.
- **`web_version/`** — Flask + HTML5 canvas web app. See `web_version/CLAUDE.md`.

Each subfolder is self-contained: its own `train_model.py`, its own copy of `digit_model.joblib` / `digit_scaler.joblib`, and (for the desktop version) its own cached `mnist_data/`. There are no shared modules between the two — logic that exists in both (notably the MNIST-style image preprocessing: crop to ink bounding box, scale into a 20x20 box, center by center of mass in a 28x28 frame) is duplicated by design and must be kept in sync manually if changed in one place. Read the relevant subfolder's `CLAUDE.md` before working in it — the details that matter (why MNIST is fetched from a mirror instead of openml.org, why the canvas preprocessing can't be a plain resize, why prediction is triggered by mouse-release rather than a button) live there, not here.

## Conventions

- All code and comments are written in English, even though user-facing chat interaction may happen in Korean.
- Every new file created in this repo should have a creation date/time comment near its top (e.g. `# Created: 2026-08-16 22:13`), per user instruction.

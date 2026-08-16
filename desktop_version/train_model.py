"""Train a handwritten digit classifier on the MNIST dataset and save it to disk.

MNIST is downloaded directly from a public mirror as raw IDX files instead of
going through sklearn's fetch_openml, since the openml.org API can be flaky.
"""

import gzip
import os
import time
import urllib.request

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "digit_model.joblib"
SCALER_PATH = "digit_scaler.joblib"
DATA_DIR = "mnist_data"

MIRROR_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            continue
        url = MIRROR_BASE + filename
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2051, f"Unexpected magic number for images: {magic}"
    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(num_images, rows * cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2049, f"Unexpected magic number for labels: {magic}"
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels


def main():
    print("Fetching MNIST dataset...")
    download_mnist_files()

    X_train = read_idx_images(os.path.join(DATA_DIR, FILES["train_images"])).astype(np.float32)
    y_train = read_idx_labels(os.path.join(DATA_DIR, FILES["train_labels"])).astype(int)
    X_test = read_idx_images(os.path.join(DATA_DIR, FILES["test_images"])).astype(np.float32)
    y_test = read_idx_labels(os.path.join(DATA_DIR, FILES["test_labels"])).astype(int)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Scale pixel values to zero mean / unit variance for faster MLP convergence.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training MLP classifier...")
    start = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        max_iter=30,
        early_stopping=True,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train, y_train)
    print(f"Training finished in {time.time() - start:.1f}s")

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test accuracy: {accuracy:.4f}")

    dump(clf, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()

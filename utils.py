"""Shared helpers: loading a bundled scikit-learn dataset and a CLI parser."""
import argparse
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split


def load_data(
        dataset: str = "iris",
        test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_test, y_train, y_test.

    iris:   150 flowers, 4 continuous features, 3 classes.
    digits: 1797 8x8 images, 64 pixel features, 10 classes.
    """
    data = load_iris() if dataset == "iris" else load_digits()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=test_size, random_state=2023)


def get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["iris", "digits"], default="iris")
    return parser


def evaluate(model, X_train, X_test, y_train, y_test, name: str = "Our") -> None:
    """Print train and test accuracy of any object with fit/predict."""
    model.fit(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"{name} train accuracy: {train_acc:.4f} | test accuracy: {test_acc:.4f}")

"""Evaluation metrics and single-image inference."""

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data import pil_loader


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true_onehot: np.ndarray,
    dataset_name: str = "Dataset",
) -> dict:
    y_true = y_true_onehot.argmax(1)
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = y_pred_probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = float("nan")

    print(f"\nEvaluation: {dataset_name}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Precision   : {prec*100:.2f}%")
    print(f"  Recall      : {rec*100:.2f}%")
    print(f"  F1 Score    : {f1*100:.2f}%")
    print(f"  Specificity : {spec*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "confusion_matrix": cm,
    }


def predict_single_image(
    model: tf.keras.Model,
    image_path: str,
    inp_dims: tuple = (224, 224, 3),
) -> dict:
    img = pil_loader(image_path)
    img = img.resize(inp_dims[:2], Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-10)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    pred_class = int(probs.argmax())
    class_names = ["Non-COVID-19", "COVID-19"]

    return {
        "prediction": class_names[pred_class],
        "covid_probability": float(probs[1]),
        "probabilities": probs.tolist(),
    }

"""Evaluation metrics.

Macro-F1 is the *primary* selection metric: it weights every class equally, so
minority classes (and the catch-all "general pathological conditions") are not
drowned out by the majority. Accuracy and weighted-F1 are reported alongside
for context.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax for a (n, num_labels) array."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Core scalar metrics from hard predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def compute_metrics_for_trainer(eval_pred) -> Dict[str, float]:
    """Adapter for HuggingFace ``Trainer(compute_metrics=...)``.

    ``eval_pred`` is a ``(predictions, labels)`` tuple where predictions are
    logits.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return classification_scores(np.asarray(labels), preds)


def confusion(
    y_true: np.ndarray, y_pred: np.ndarray, num_labels: int
) -> np.ndarray:
    """Confusion matrix with a fixed label ordering ``0..num_labels-1``."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """F1 broken out per class — useful for error analysis in the demo."""
    num_labels = len(label_names) if label_names else int(max(y_true.max(), y_pred.max())) + 1
    scores = f1_score(
        y_true, y_pred, labels=list(range(num_labels)), average=None, zero_division=0
    )
    if label_names:
        return {name: float(s) for name, s in zip(label_names, scores)}
    return {str(i): float(s) for i, s in enumerate(scores)}

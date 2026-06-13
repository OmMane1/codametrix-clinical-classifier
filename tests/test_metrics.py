"""Tests for metric computation."""

import numpy as np

from encoder.metrics import (
    classification_scores,
    compute_metrics_for_trainer,
    confusion,
    per_class_f1,
    softmax,
)


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Uniform logits -> uniform probabilities.
    assert np.allclose(probs[1], 1 / 3)


def test_softmax_is_numerically_stable_for_large_logits():
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs = softmax(logits)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(), 1.0)


def test_classification_scores_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4])
    scores = classification_scores(y, y)
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["weighted_f1"] == 1.0


def test_compute_metrics_for_trainer_uses_argmax():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])  # -> preds [1, 0]
    labels = np.array([1, 0])
    scores = compute_metrics_for_trainer((logits, labels))
    assert scores["accuracy"] == 1.0


def test_per_class_f1_returns_named_keys():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    names = ["a", "b", "c"]
    out = per_class_f1(y_true, y_pred, names)
    assert set(out) == {"a", "b", "c"}
    assert out["a"] == 1.0
    assert out["c"] == 0.0  # class c never predicted correctly


def test_confusion_has_fixed_shape():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    cm = confusion(y_true, y_pred, num_labels=5)
    assert cm.shape == (5, 5)

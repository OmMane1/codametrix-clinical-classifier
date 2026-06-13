"""Tests for the submission-format builder (the part that must be exactly right).

``encoder.predict`` has no module-level torch import, so this is testable on CPU.
"""

import numpy as np

from encoder.predict import build_submission_frames

LABELS = ["cardiology", "neurology", "orthopedics", "gastroenterology", "other"]

# Three cases: argmax -> class 0, 4, 2
PROBS = np.array(
    [
        [0.7, 0.1, 0.05, 0.05, 0.1],
        [0.1, 0.1, 0.0, 0.1, 0.7],
        [0.1, 0.2, 0.6, 0.05, 0.05],
    ]
)
CASE_IDS = [1, 2, 3]


def test_submission_has_exactly_two_columns_in_order():
    sub, _ = build_submission_frames(CASE_IDS, PROBS, LABELS)
    assert list(sub.columns) == ["case_number", "prediction"]


def test_submission_predicts_class_names_by_default():
    sub, _ = build_submission_frames(CASE_IDS, PROBS, LABELS)
    assert list(sub["case_number"]) == [1, 2, 3]
    assert list(sub["prediction"]) == ["cardiology", "other", "orthopedics"]


def test_submission_label_ids_zero_based():
    sub, _ = build_submission_frames(CASE_IDS, PROBS, LABELS, label_as_id=True)
    assert list(sub["prediction"]) == [0, 4, 2]


def test_submission_label_ids_one_based():
    sub, _ = build_submission_frames(
        CASE_IDS, PROBS, LABELS, label_as_id=True, one_based_output=True
    )
    assert list(sub["prediction"]) == [1, 5, 3]


def test_detailed_frame_has_per_class_probabilities():
    _, detailed = build_submission_frames(CASE_IDS, PROBS, LABELS)
    expected = ["case_number", "prediction"] + [f"prob_{n}" for n in LABELS]
    assert list(detailed.columns) == expected
    # row 0 probability for cardiology matches the input
    assert detailed.loc[0, "prob_cardiology"] == 0.7

"""Data loading, parsing and fold assignment for the encoder.

Two distinct input formats are handled:

1. **Training corpus** (`train.dat`, the Kaggle "medical-text" dataset):
   tab-separated lines of ``<label>\\t<text>`` with 1-based integer labels
   ``1..5``. We convert to 0-based indices internally.

2. **Hidden test file** (sent by the organiser): plain text split into blocks::

       Case 1
       <one or more lines of abstract text>

       Case 2
       <more text>

   Each block is one example to classify. The parser is tolerant of blank
   lines, multi-line case bodies and trailing whitespace.

Text *cleaning* is intentionally out of scope here — a teammate owns
``data_cleaning.py``. This module accepts an optional ``cleaner`` callable so
the same cleaning is applied consistently to train and test, but defaults to a
no-op identity so the encoder pipeline is usable on its own.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Matches a "Case 12" header line (case-insensitive), optional surrounding
# whitespace, nothing else on the line.
_CASE_HEADER_RE = re.compile(r"^\s*case\s+(\d+)\s*$", re.IGNORECASE)

# Identity cleaner used when the caller does not supply one.
def _identity(text: str) -> str:
    return text


def load_training_data(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    cleaner: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Load the tab-separated training corpus into a tidy DataFrame.

    Returns a DataFrame with columns ``[text_column, label_column, "raw_label"]``
    where ``label_column`` is 0-based and ``raw_label`` preserves the original
    1-based value for traceability.

    Raises ``ValueError`` if labels fall outside the expected ``1..5`` range,
    which would silently corrupt training.
    """
    cleaner = cleaner or _identity
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    # The corpus has no header. ``label<TAB>text``; text may itself contain
    # tabs, so split on the FIRST tab only.
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected '<label>\\t<text>', got: {line[:80]!r}"
                )
            raw_label_str, text = parts
            try:
                raw_label = int(raw_label_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: label is not an integer: {raw_label_str!r}"
                ) from exc
            rows.append({"raw_label": raw_label, text_column: cleaner(text.strip())})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable rows parsed from {path}")

    bad = df.loc[(df["raw_label"] < 1) | (df["raw_label"] > 5), "raw_label"].unique()
    if len(bad):
        raise ValueError(
            f"Found labels outside 1..5 in {path}: {sorted(bad)}. "
            "The medical-text corpus uses 1-based labels 1..5."
        )

    df[label_column] = df["raw_label"] - 1  # 0-based for the model
    return df.reset_index(drop=True)


def load_clean_csv(
    path: str | Path,
    text_column: str = "encoder_text",
    label_column: str = "hackathon_label",
    num_labels: int = 5,
    drop_duplicate_text: bool = True,
    min_text_chars: int = 1,
    cleaner: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Load the cleaned CSV produced by ``data_cleaning.py``.

    This is the encoder's primary input. It uses the ``encoder_text`` column
    (lightly normalised, casing/digits/punctuation preserved — best for a
    transformer) and the ``hackathon_label`` column (the remapped 0..4 class
    space), exactly as recommended in ``data-cleaning-notes.md``.

    Steps: select the two columns, drop empty/near-empty text, validate the
    label range, and (optionally) drop duplicate texts so the same note cannot
    leak across CV folds.

    Returns a DataFrame with columns ``[text_column, label_column]``.
    """
    cleaner = cleaner or _identity
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {path}")

    df = pd.read_csv(path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise KeyError(
            f"{path} is missing required column(s) {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )

    df = df[[text_column, label_column]].copy()
    df[text_column] = df[text_column].fillna("").astype(str).map(cleaner).str.strip()

    before = len(df)
    df = df[df[text_column].str.len() >= min_text_chars]
    if len(df) < before:
        # Empty rows are dropped silently here but reported by the caller via
        # the row count; flag if it is surprising.
        pass

    try:
        df[label_column] = df[label_column].astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: {label_column} is not integer-coercible.") from exc

    bad = df.loc[
        (df[label_column] < 0) | (df[label_column] >= num_labels), label_column
    ].unique()
    if len(bad):
        raise ValueError(
            f"{path}: {label_column} has values outside 0..{num_labels - 1}: "
            f"{sorted(bad)}"
        )

    if drop_duplicate_text:
        df = df.drop_duplicates(subset=text_column)

    if df.empty:
        raise ValueError(f"No usable rows after filtering {path}")
    return df.reset_index(drop=True)


def compute_class_weights(labels, num_labels: int) -> np.ndarray:
    """Inverse-frequency ("balanced") class weights over *present* classes.

    Matches sklearn's ``class_weight="balanced"`` for classes that appear, and
    assigns weight ``1.0`` to absent classes (e.g. ``orthopedics``, which has no
    training rows in this corpus) so they neither break the computation nor
    distort the loss. Pure NumPy — no torch — so it is unit-testable.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    weights = np.ones(num_labels, dtype=np.float64)
    present = counts > 0
    n_present_classes = int(present.sum())
    n_samples = counts[present].sum()
    if n_present_classes > 0:
        weights[present] = n_samples / (n_present_classes * counts[present])
    return weights


def parse_case_file(
    path: str | Path,
    text_column: str = "text",
    cleaner: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Parse the organiser's ``Case N`` hidden-test file.

    Returns a DataFrame with columns ``["case_id", text_column]`` ordered by
    appearance. ``case_id`` is the integer parsed from each ``Case N`` header.

    The parser raises ``ValueError`` if it finds body text before the first
    ``Case`` header, or if a case ends up with empty text — both signal a
    malformed file we would rather fail loudly on than submit garbage.
    """
    cleaner = cleaner or _identity
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    cases: List[dict] = []
    current_id: Optional[int] = None
    buffer: List[str] = []

    def _flush() -> None:
        if current_id is None:
            return
        body = cleaner(" ".join(buffer).strip())
        if not body:
            raise ValueError(f"Case {current_id} in {path} has empty body text.")
        cases.append({"case_id": current_id, text_column: body})

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            header = _CASE_HEADER_RE.match(line)
            if header:
                _flush()  # close the previous case
                current_id = int(header.group(1))
                buffer = []
            else:
                stripped = line.strip()
                if not stripped:
                    continue
                if current_id is None:
                    raise ValueError(
                        f"{path}:{line_no}: body text before any 'Case N' header: "
                        f"{stripped[:80]!r}"
                    )
                buffer.append(stripped)
    _flush()  # close the final case

    if not cases:
        raise ValueError(f"No 'Case N' blocks found in {path}")

    df = pd.DataFrame(cases)
    dupes = df["case_id"][df["case_id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"Duplicate case ids in {path}: {sorted(dupes)}")
    return df.reset_index(drop=True)


def add_stratified_folds(
    df: pd.DataFrame,
    label_column: str = "label",
    n_folds: int = 5,
    seed: int = 2024,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Assign each row a fold index via stratified K-fold.

    Returns a copy with an integer ``fold_column``. Stratifying on the label
    keeps class proportions stable across folds — important given class
    imbalance in this corpus.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    df = df.copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df[fold_column] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df[label_column])):
        df.loc[df.index[val_idx], fold_column] = fold

    assert (df[fold_column] >= 0).all(), "Every row must be assigned a fold."
    return df

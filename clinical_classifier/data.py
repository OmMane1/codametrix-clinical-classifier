"""Data loading.

Handles two input shapes so the same code works for the raw Kaggle dump and
the cleaned hackathon CSV:

  * ``.csv`` — uses ``text_col`` / ``label_col`` (label_col optional for test).
  * ``.dat`` / ``.txt`` — Kaggle `medical-text` format: each line is
    ``<int label><whitespace><abstract text>``. For an unlabeled test file the
    leading integer is simply absent and the whole line is treated as text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from . import config

# A line that starts with a small standalone integer => that integer is the label.
_LABELED_DAT = re.compile(r"^\s*(\d+)\s+(.*\S.*)$")


def _clean(text: str) -> str:
    """Light normalization — collapse whitespace. (TF-IDF does the rest.)"""
    return re.sub(r"\s+", " ", str(text)).strip()


def _load_dat(path: Path) -> Tuple[List[str], Optional[List[str]]]:
    texts, labels = [], []
    any_label = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        m = _LABELED_DAT.match(raw)
        if m:
            any_label = True
            labels.append(m.group(1))
            texts.append(_clean(m.group(2)))
        else:
            labels.append(None)
            texts.append(_clean(raw))
    return texts, (labels if any_label else None)


def load_dataset(
    path,
    text_col: str = config.DEFAULT_TEXT_COL,
    label_col: Optional[str] = config.DEFAULT_LABEL_COL,
    label_map: Optional[dict] = None,
    require_labels: bool = True,
) -> Tuple[List[str], Optional[List[str]]]:
    """Return ``(texts, labels)``. ``labels`` is ``None`` for unlabeled inputs.

    Args:
        label_map: optional dict applied to each raw label (e.g. Kaggle numeric
            -> hackathon name). Unmapped labels are kept as-is.
        require_labels: raise if labels are expected but missing.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if text_col not in df.columns:
            raise KeyError(
                f"text column {text_col!r} not in {path.name}; columns={list(df.columns)}"
            )
        texts = [_clean(t) for t in df[text_col].fillna("")]
        labels = None
        if label_col and label_col in df.columns:
            labels = [None if pd.isna(v) else str(v) for v in df[label_col]]
    else:
        texts, labels = _load_dat(path)

    if label_map and labels is not None:
        labels = [label_map.get(l, l) for l in labels]

    if require_labels and (labels is None or all(l is None for l in labels)):
        raise ValueError(
            f"No labels found in {path}. Pass require_labels=False for a test file."
        )
    return texts, labels


def kaggle_bootstrap(path) -> Tuple[List[str], List[str]]:
    """Convenience loader: raw Kaggle .dat -> hackathon label names."""
    texts, labels = load_dataset(path, label_map=config.KAGGLE_LABEL_MAP)
    return texts, labels

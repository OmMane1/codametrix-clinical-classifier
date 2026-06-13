"""Inference on the organiser's ``Case N`` hidden-test file.

Loads every fold model trained by :mod:`encoder.train`, runs each over the
parsed test cases, and **averages the softmax probabilities** across folds
(a simple, robust ensemble). Writes:

* ``<run>_submission.csv`` — the official 2-column submission:
  ``case_number, prediction`` (one row per case).
* ``<run>_submission_detailed.csv`` — case_number + predicted name + per-class
  probabilities, for error analysis / the demo (not for submission).
* ``<run>_test_probs.npy`` — the averaged ``(n_cases, num_labels)`` matrix,
  so the team ensemble can combine it with the TF-IDF model's test probs.

The ``prediction`` column is the class *name* (e.g. ``cardiology``) by default.
Set ``label_as_id=True`` to emit the numeric ``hackathon_label`` instead
(0-based; ``one_based_output=True`` makes it 1-based).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .config import EncoderConfig
from .data import parse_case_file
from .metrics import softmax

logger = logging.getLogger(__name__)


def build_submission_frames(
    case_ids,
    avg_probs: np.ndarray,
    label_names: List[str],
    label_as_id: bool = False,
    one_based_output: bool = False,
):
    """Build the (official, detailed) submission DataFrames from probabilities.

    Pure function (no torch / no model) so the exact output format is
    unit-testable. Returns ``(submission_df, detailed_df)`` where:

    * ``submission_df`` has exactly two columns: ``case_number, prediction``.
    * ``detailed_df`` adds per-class probability columns for error analysis.
    """
    pred_idx = avg_probs.argmax(axis=1)
    pred_names = [label_names[i] for i in pred_idx]

    if label_as_id:
        predictions = pred_idx + (1 if one_based_output else 0)
    else:
        predictions = pred_names

    submission = pd.DataFrame({"case_number": list(case_ids), "prediction": predictions})

    detailed = pd.DataFrame({"case_number": list(case_ids), "prediction": pred_names})
    for i, name in enumerate(label_names):
        detailed[f"prob_{name}"] = avg_probs[:, i]
    return submission, detailed


def _discover_fold_dirs(config: EncoderConfig) -> List[Path]:
    """Find saved fold model directories for this run."""
    root = Path(config.output_dir)
    dirs = sorted(
        d for d in root.glob(f"{config.run_name}_fold*") if (d / "config.json").exists()
    )
    if not dirs:
        raise FileNotFoundError(
            f"No fold models found under {root} for run '{config.run_name}'. "
            "Train first."
        )
    return dirs


def predict_test(
    config: EncoderConfig,
    test_path: str | Path,
    cleaner: Optional[Callable[[str], str]] = None,
    one_based_output: bool = False,
    label_as_id: bool = False,
) -> pd.DataFrame:
    """Parse the test file, ensemble fold models, and write a submission.

    Returns the 2-column submission DataFrame (``case_number, prediction``).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cases = parse_case_file(test_path, text_column=config.text_column, cleaner=cleaner)
    texts = cases[config.text_column].tolist()
    logger.info("Parsed %d test cases from %s", len(cases), test_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_dirs = _discover_fold_dirs(config)
    logger.info("Ensembling %d fold model(s) on %s", len(fold_dirs), device)

    summed = np.zeros((len(texts), config.num_labels), dtype=np.float64)

    for fold_dir in fold_dirs:
        tokenizer = AutoTokenizer.from_pretrained(str(fold_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(fold_dir)).to(device)
        model.eval()

        fold_logits = np.zeros((len(texts), config.num_labels), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), config.eval_batch_size):
                batch = texts[start : start + config.eval_batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    max_length=config.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits.detach().cpu().numpy()
                fold_logits[start : start + len(batch)] = logits

        summed += softmax(fold_logits)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    avg_probs = summed / len(fold_dirs)

    sub_dir = Path(config.submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / f"{config.run_name}_test_probs.npy", avg_probs.astype(np.float32))

    submission, detailed = build_submission_frames(
        cases["case_id"],
        avg_probs,
        config.label_names,
        label_as_id=label_as_id,
        one_based_output=one_based_output,
    )
    out_csv = sub_dir / f"{config.run_name}_submission.csv"
    submission.to_csv(out_csv, index=False)
    detailed.to_csv(sub_dir / f"{config.run_name}_submission_detailed.csv", index=False)

    logger.info("Wrote %s (2-col submission) + detailed CSV + test probs.", out_csv)
    return submission

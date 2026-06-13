"""Load a saved model and write predictions for the hidden test set to CSV.

Example
-------
python -m clinical_classifier.predict \
    --model models/tfidf_logreg.joblib \
    --data data/raw/test.dat \
    --out submission.csv

Output columns: id, text, prediction  (+ per-class probabilities with --proba).
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from . import config
from .data import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Predict clinical note categories -> CSV.")
    p.add_argument("--model", default=str(config.MODELS_DIR / "tfidf_logreg.joblib"))
    p.add_argument("--data", required=True, help="Path to test .csv or .dat (unlabeled).")
    p.add_argument("--text-col", default=config.DEFAULT_TEXT_COL)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--proba", action="store_true", help="Include per-class probabilities.")
    p.add_argument("--keep-text", action="store_true", help="Include note text in output.")
    return p.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    texts, _ = load_dataset(
        args.data, text_col=args.text_col, label_col=None, require_labels=False,
    )

    # preserve the source `id` (so predictions join back to truth / the test set)
    ids = list(range(len(texts)))
    if str(args.data).lower().endswith(".csv"):
        raw = pd.read_csv(args.data)
        if "id" in raw.columns:
            ids = raw["id"].tolist()

    preds = model.predict(texts)
    out = pd.DataFrame({"id": ids, "prediction": preds})
    if args.keep_text:
        out["text"] = texts

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        for i, cls in enumerate(model.classes_):
            out[f"proba_{cls}"] = proba[:, i]

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions -> {args.out}")
    print(out["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()

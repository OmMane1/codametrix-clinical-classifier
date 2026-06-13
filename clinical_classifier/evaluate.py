"""Shared evaluation — the ONE scorer the whole team uses.

Two ways to use it:

1. As a library, on in-memory predictions:
       from clinical_classifier.evaluate import evaluate
       evaluate(y_true, y_pred)          # prints report, returns dict

2. As a CLI, scoring a predictions file against the frozen val set:
       python -m clinical_classifier.evaluate \
           --pred my_model_val_preds.csv --truth data/splits/val.csv

   Your predictions CSV must have columns `id` and `prediction`, where `id`
   matches the `id` in val.csv. macro-F1 is the headline metric (every class
   weighted equally, so the big "Other" bucket can't mask weak classes).
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(y_true, y_pred, *, labels=None, title="", show=True) -> dict:
    """Score predictions. Returns a dict; prints a human report when show=True."""
    if labels is None:
        labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    if show:
        if title:
            print(f"\n=== {title} ===")
        print(f"macro-F1: {macro_f1:.4f}   accuracy: {acc:.4f}\n")
        print(classification_report(y_true, y_pred, labels=labels, digits=3,
                                    zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        head = "true\\pred  " + " ".join(f"{l[:6]:>7}" for l in labels)
        print(head)
        for lab, row in zip(labels, cm):
            print(f"{lab[:9]:<10} " + " ".join(f"{v:>7}" for v in row))

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "labels": labels,
        "report": classification_report(y_true, y_pred, labels=labels,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description="Score predictions against the frozen val set.")
    p.add_argument("--pred", required=True, help="CSV with columns id, prediction")
    p.add_argument("--truth", default="data/splits/val.csv",
                   help="frozen val CSV with columns id, label")
    args = p.parse_args()

    pred = pd.read_csv(args.pred)
    truth = pd.read_csv(args.truth)
    for col, df, name in [("prediction", pred, args.pred), ("label", truth, args.truth)]:
        if col not in df.columns or "id" not in df.columns:
            raise KeyError(f"{name} must have columns 'id' and '{col}'; got {list(df.columns)}")

    merged = truth[["id", "label"]].merge(pred[["id", "prediction"]], on="id", how="left")
    missing = merged["prediction"].isna().sum()
    if missing:
        raise ValueError(f"{missing} val rows have no prediction (id mismatch?)")

    evaluate(merged["label"].tolist(), merged["prediction"].tolist(),
             title=f"{args.pred} vs {args.truth}")


if __name__ == "__main__":
    main()

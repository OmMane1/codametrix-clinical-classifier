"""Train the TF-IDF baseline, report cross-validated metrics, save the model.

Examples
--------
# Bootstrap on the raw Kaggle dump (numeric labels -> hackathon names):
python -m clinical_classifier.train --data data/raw/train.dat --kaggle-map

# Train on the cleaned hackathon CSV:
python -m clinical_classifier.train --data data/train.csv --text-col text --label-col label

# Add a (slower) grid search over TF-IDF + C:
python -m clinical_classifier.train --data data/train.csv --search
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config
from .data import load_dataset
from .pipeline import PARAM_GRID, build_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Train TF-IDF clinical note classifier.")
    p.add_argument("--data", required=True, help="Path to training .csv or .dat")
    p.add_argument("--text-col", default=config.DEFAULT_TEXT_COL)
    p.add_argument("--label-col", default=config.DEFAULT_LABEL_COL)
    p.add_argument("--kaggle-map", action="store_true",
                   help="Map raw Kaggle numeric labels -> hackathon names.")
    p.add_argument("--clf", default="logreg", choices=["logreg", "linearsvc"],
                   help="Classifier head (default: logreg).")
    p.add_argument("--search", action="store_true", help="Run GridSearchCV.")
    p.add_argument("--cv", type=int, default=5, help="CV folds (default 5).")
    p.add_argument("--no-char", action="store_true", help="Disable char n-grams.")
    p.add_argument("--out", default=str(config.MODELS_DIR / "tfidf_logreg.joblib"))
    return p.parse_args()


def main():
    args = parse_args()
    label_map = config.KAGGLE_LABEL_MAP if args.kaggle_map else None
    texts, labels = load_dataset(
        args.data, text_col=args.text_col, label_col=args.label_col,
        label_map=label_map,
    )
    X = np.array(texts, dtype=object)
    y = np.array(labels)
    print(f"Loaded {len(X)} notes | classes: {sorted(set(y))}")
    for cls, n in zip(*np.unique(y, return_counts=True)):
        print(f"  {cls:<18} {n}")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=config.RANDOM_STATE)

    if args.search:
        base = build_pipeline(clf=args.clf, use_char=not args.no_char)
        search = GridSearchCV(
            base, PARAM_GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)
        print(f"\nBest CV macro-F1: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        model = search.best_estimator_
    else:
        model = build_pipeline(clf=args.clf, use_char=not args.no_char)

    # Honest, leakage-free metrics via cross-validated predictions on all data.
    print("\nComputing cross-validated predictions for reporting...")
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    macro_f1 = f1_score(y, y_pred, average="macro")
    print(f"\nCross-validated macro-F1: {macro_f1:.4f}\n")
    print(classification_report(y, y_pred, digits=3))

    # Refit on ALL data before saving (more data -> better final model).
    model.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved model -> {args.out}")

    labels_sorted = sorted(set(y))
    report = {
        "cv_macro_f1": macro_f1,
        "classes": labels_sorted,
        "classification_report": classification_report(
            y, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels_sorted).tolist(),
    }
    report_path = config.REPORTS_DIR / "cv_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved CV report -> {report_path}")


if __name__ == "__main__":
    main()

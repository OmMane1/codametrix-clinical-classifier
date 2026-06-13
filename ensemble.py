"""Ensemble two (or more) models by weighted-averaging their class probabilities.

Each input CSV must have an `id` column plus `proba_<Class>` columns (the format
`clinical_classifier.predict --proba` emits). All models must use the SAME class
names and the SAME `id`s (e.g. both predict on data/splits/val.csv). Rows are
aligned by `id`; classes by name — order doesn't matter.

Tune the weight on the frozen val set:
    python ensemble.py --proba tfidf_val_proba.csv transformer_val_proba.csv \
        --truth data/splits/val.csv --sweep

Produce the final submission on the hidden test:
    python ensemble.py --proba tfidf_test_proba.csv transformer_test_proba.csv \
        --weights 0.4 0.6 --out submission.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def load_proba(path):
    """-> DataFrame indexed by id, columns = class names (proba_ prefix stripped)."""
    df = pd.read_csv(path).set_index("id")
    cols = [c for c in df.columns if c.startswith("proba_")]
    if not cols:
        raise ValueError(f"{path} has no proba_<Class> columns; got {list(df.columns)}")
    out = df[cols].copy()
    out.columns = [c[len("proba_"):] for c in cols]
    return out


def combine(probas, weights):
    """Weighted-average aligned probabilities. Returns DataFrame (id x class)."""
    ids = probas[0].index
    for p in probas[1:]:
        ids = ids.intersection(p.index)
    classes = sorted(set.intersection(*[set(p.columns) for p in probas]))
    if len(ids) == 0 or not classes:
        raise ValueError("no common ids/classes across inputs — check id alignment & class names")
    wsum = float(sum(weights))
    acc = None
    for p, w in zip(probas, weights):
        part = p.loc[ids, classes].to_numpy() * w
        acc = part if acc is None else acc + part
    return pd.DataFrame(acc / wsum, index=ids, columns=classes)


def predictions(combined):
    return combined.idxmax(axis=1)


def main():
    ap = argparse.ArgumentParser(description="Probability-average ensemble.")
    ap.add_argument("--proba", nargs="+", required=True, help="2+ proba CSVs")
    ap.add_argument("--weights", nargs="*", type=float, help="one per model (default equal)")
    ap.add_argument("--truth", help="val CSV with id,label — enables scoring")
    ap.add_argument("--sweep", action="store_true", help="(2 models) sweep weight, report best macro-F1")
    ap.add_argument("--out", help="write ensembled id,prediction,proba_* to this CSV")
    args = ap.parse_args()

    probas = [load_proba(p) for p in args.proba]
    weights = args.weights or [1.0] * len(probas)
    if len(weights) != len(probas):
        ap.error("--weights must have one value per --proba file")

    if args.sweep:
        if len(probas) != 2:
            ap.error("--sweep supports exactly 2 models")
        if not args.truth:
            ap.error("--sweep needs --truth")
        from clinical_classifier.evaluate import evaluate
        truth = pd.read_csv(args.truth).set_index("id")["label"]
        best = None
        print(f"{'w(model1)':>9} {'macro-F1':>9}")
        for w in np.linspace(0, 1, 21):
            comb = combine(probas, [w, 1 - w])
            y_pred = predictions(comb)
            ids = comb.index
            r = evaluate(truth.loc[ids].tolist(), y_pred.loc[ids].tolist(), show=False)
            print(f"{w:>9.2f} {r['macro_f1']:>9.4f}")
            if best is None or r["macro_f1"] > best[1]:
                best = (w, r["macro_f1"])
        print(f"\nbest: w(model1)={best[0]:.2f}, 1-w={1-best[0]:.2f} -> macro-F1 {best[1]:.4f}")
        print("(model1 = first --proba file; reuse via --weights)")
        return

    comb = combine(probas, weights)
    y_pred = predictions(comb)

    if args.truth:
        from clinical_classifier.evaluate import evaluate
        truth = pd.read_csv(args.truth).set_index("id")["label"]
        ids = comb.index
        evaluate(truth.loc[ids].tolist(), y_pred.loc[ids].tolist(),
                 title=f"ensemble weights={weights}")

    if args.out:
        out = comb.copy()
        out.columns = [f"proba_{c}" for c in out.columns]
        out.insert(0, "prediction", y_pred)
        out.reset_index().to_csv(args.out, index=False)
        print(f"wrote {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()

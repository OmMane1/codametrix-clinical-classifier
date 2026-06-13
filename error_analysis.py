"""Explained error analysis on the frozen val set.

Uses the baseline trained on the TRAIN split (no leakage) to predict VAL, then
for the most common confusions shows example notes WITH their word-level
explanation — i.e. *why* the model slipped. Useful for the writeup and for
telling teammates where a transformer / ensemble should help.

Usage:  python error_analysis.py
"""
from collections import Counter

import joblib
import pandas as pd

from clinical_classifier.explain import token_contributions

MODEL = "models/tfidf_baseline.joblib"   # trained on the train split only


def main():
    model = joblib.load(MODEL)
    val = pd.read_csv("data/splits/val.csv")
    val["pred"] = model.predict(val["text"])

    wrong = val[val["pred"] != val["label"]]
    print(f"val: {len(val)} notes | errors: {len(wrong)} (acc {1-len(wrong)/len(val):.3f})\n")

    confusions = Counter(zip(wrong["label"], wrong["pred"]))
    print("Most common confusions (true -> pred):")
    for (t, p), n in confusions.most_common(6):
        print(f"  {n:>3}  {t} -> {p}")

    # drill into the top confusion with explanations
    (t, p), _ = confusions.most_common(1)[0]
    print(f"\n=== Why '{t}' notes get called '{p}' (up to 3 examples) ===")
    for _, row in wrong[(wrong.label == t) & (wrong.pred == p)].head(3).iterrows():
        pred, proba, contrib = token_contributions(model, row["text"])
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:8]
        print(f"\n  note[:110]: {row['text'][:110]!r}")
        print(f"  predicted {pred}: " + ", ".join(f"{c} {q:.2f}" for c, q in list(proba.items())[:3]))
        print(f"  drove '{pred}': " + ", ".join(f"{w}({v:+.2f})" for w, v in top))


if __name__ == "__main__":
    main()

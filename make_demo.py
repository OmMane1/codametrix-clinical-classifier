"""Generate a standalone HTML page with explained predictions (no server needed).

Open the output in any browser — good for the slide deck or sharing. Includes:
  - what the model learned each specialty looks like (global terms),
  - one confident example per class (auto-route),
  - the single lowest-confidence val note (flagged for human review).

Usage:  python make_demo.py
"""
import os

import joblib
import numpy as np
import pandas as pd

from clinical_classifier.explain import global_top_terms, render_html

# Baseline model is trained on the TRAIN split only, so its predictions on the
# val notes below are genuine held-out (not in-sample) — honest + surfaces the
# real low-confidence cases for the human-review story.
MODEL = "models/tfidf_baseline.joblib"
OUT = "demo/explain_demo.html"


def main():
    os.makedirs("demo", exist_ok=True)
    model = joblib.load(MODEL)
    val = pd.read_csv("data/splits/val.csv")

    # global vocabulary table
    vocab = "".join(
        f'<tr><td style="padding:4px 12px;font-weight:600">{cls}</td>'
        f'<td style="padding:4px 12px;color:#444">{", ".join(t for t, _ in terms)}</td></tr>'
        for cls, terms in global_top_terms(model, n=12).items())

    blocks = []
    # one confident example per class
    for lab in ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]:
        sub = val[val.label == lab]
        if sub.empty:
            continue
        note = sub.iloc[0]["text"][:1200]
        blocks.append(f'<p style="color:#888;font-family:system-ui;margin-top:24px">'
                      f'true label: <b>{lab}</b></p>' + render_html(model, note))

    # the lowest-confidence note -> shows the "flag for review" routing
    conf = model.predict_proba(val["text"]).max(axis=1)
    low = val.iloc[int(np.argmin(conf))]
    blocks.append('<h2 style="font-family:system-ui;margin-top:32px">Hardest case '
                  '(lowest confidence → human review)</h2>'
                  f'<p style="color:#888;font-family:system-ui">true label: '
                  f'<b>{low["label"]}</b></p>' + render_html(model, low["text"][:1200]))

    page = (
        "<html><head><meta charset='utf-8'><title>Explainable Clinical Classifier</title></head>"
        "<body style='background:#fafafa;margin:0;padding:24px'>"
        "<h1 style='font-family:system-ui'>Explainable Clinical Note Classification</h1>"
        "<p style='font-family:system-ui;color:#555;max-width:820px'>TF-IDF + Logistic "
        "Regression. Each prediction decomposes <i>exactly</i> into per-word contributions, "
        "and low-confidence notes are routed to a human — auditable and deployable. "
        "<i>(Predictions below are on held-out validation notes.)</i></p>"
        "<h2 style='font-family:system-ui'>What the model learned</h2>"
        f"<table style='font-family:system-ui;font-size:14px;border-collapse:collapse'>{vocab}</table>"
        "<h2 style='font-family:system-ui;margin-top:32px'>Per-note explanations</h2>"
        + "".join(blocks) + "</body></html>")

    with open(OUT, "w") as f:
        f.write(page)
    print(f"wrote {OUT}  ({len(blocks)} examples, lowest conf = {conf.min():.0%})")


if __name__ == "__main__":
    main()

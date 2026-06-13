"""Generate a standalone HTML page with explained predictions (no server needed).

TF-IDF only by default. If `transformer_val_proba.csv` (from the Colab ensemble
run) is present, predictions become the ENSEMBLE blend; word highlights stay
from the interpretable TF-IDF component.

Usage:  python make_demo.py
"""
import os

import joblib
import numpy as np
import pandas as pd

from clinical_classifier.explain import (
    ensemble_proba,
    global_top_terms,
    render_html,
    render_html_ensemble,
)

MODEL = "models/tfidf_baseline.joblib"           # held-out model (matches val)
TRANSFORMER_PROBA = "transformer_val_proba.csv"  # optional, from Colab
W_TFIDF = 0.40                                   # set to the best weight from ensemble.py --sweep
OUT = "demo/explain_demo.html"


def load_tproba():
    if not os.path.exists(TRANSFORMER_PROBA):
        return None
    df = pd.read_csv(TRANSFORMER_PROBA).set_index("id")
    cols = {c: c[len("proba_"):] for c in df.columns if c.startswith("proba_")}
    return {i: {cls: float(row[c]) for c, cls in cols.items()} for i, row in df.iterrows()}


def main():
    os.makedirs("demo", exist_ok=True)
    model = joblib.load(MODEL)
    val = pd.read_csv("data/splits/val.csv")
    tproba = load_tproba()
    mode = f"ENSEMBLE ({int(W_TFIDF*100)}% TF-IDF + {int((1-W_TFIDF)*100)}% transformer)" if tproba else "TF-IDF only"

    def render(text, note_id):
        if tproba is not None and note_id in tproba:
            return render_html_ensemble(model, text, tproba[note_id], W_TFIDF)
        return render_html(model, text)

    vocab = "".join(
        f'<tr><td style="padding:4px 12px;font-weight:600">{cls}</td>'
        f'<td style="padding:4px 12px;color:#444">{", ".join(t for t, _ in terms)}</td></tr>'
        for cls, terms in global_top_terms(model, n=12).items())

    blocks = []
    for lab in ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]:
        sub = val[val.label == lab]
        if sub.empty:
            continue
        r = sub.iloc[0]
        blocks.append(f'<p style="color:#888;font-family:system-ui;margin-top:24px">'
                      f'true label: <b>{lab}</b></p>' + render(r["text"][:1200], int(r["id"])))

    # lowest-confidence note (by the active mode) -> human-review example
    P = model.predict_proba(val["text"])
    classes = list(model.classes_)
    confs = []
    for k, (_, row) in enumerate(val.iterrows()):
        if tproba is not None and int(row["id"]) in tproba:
            tfidf_d = {c: P[k][j] for j, c in enumerate(classes)}
            confs.append(max(ensemble_proba(tfidf_d, tproba[int(row["id"])], W_TFIDF).values()))
        else:
            confs.append(float(P[k].max()))
    low = val.iloc[int(np.argmin(confs))]
    blocks.append('<h2 style="font-family:system-ui;margin-top:32px">Hardest case '
                  '(lowest confidence → human review)</h2>'
                  f'<p style="color:#888;font-family:system-ui">true label: '
                  f'<b>{low["label"]}</b></p>' + render(low["text"][:1200], int(low["id"])))

    page = (
        "<html><head><meta charset='utf-8'><title>Explainable Clinical Classifier</title></head>"
        "<body style='background:#fafafa;margin:0;padding:24px'>"
        "<h1 style='font-family:system-ui'>Explainable Clinical Note Classification</h1>"
        f"<p style='font-family:system-ui;color:#555;max-width:820px'>Mode: <b>{mode}</b>. "
        "Decision + confidence from the model(s); per-word evidence from the interpretable "
        "TF-IDF component. Low-confidence notes route to a human. "
        "<i>(Predictions on held-out validation notes.)</i></p>"
        "<h2 style='font-family:system-ui'>What the model learned</h2>"
        f"<table style='font-family:system-ui;font-size:14px;border-collapse:collapse'>{vocab}</table>"
        "<h2 style='font-family:system-ui;margin-top:32px'>Per-note explanations</h2>"
        + "".join(blocks) + "</body></html>")

    with open(OUT, "w") as f:
        f.write(page)
    print(f"wrote {OUT}  | mode: {mode} | lowest conf {min(confs):.0%}")


if __name__ == "__main__":
    main()

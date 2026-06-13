"""Generate a standalone HTML page with several explained predictions.

Open the output in any browser — no server / Streamlit needed. Good for the
slide deck or for sharing with teammates/judges.

Usage:  python make_demo.py
"""
import joblib
import pandas as pd

from clinical_classifier.explain import render_html

MODEL = "models/tfidf_submission.joblib"
OUT = "demo/explain_demo.html"


def main():
    import os
    os.makedirs("demo", exist_ok=True)
    model = joblib.load(MODEL)
    val = pd.read_csv("data/splits/val.csv")

    # one example per class (first val note of each)
    blocks = []
    for lab in ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]:
        sub = val[val.label == lab]
        if sub.empty:
            continue
        note = sub.iloc[0]["text"][:1200]  # keep the page readable
        blocks.append(f'<hr><p style="color:#888;font-family:system-ui">'
                      f'true label: <b>{lab}</b></p>' + render_html(model, note))

    page = ("<html><head><meta charset='utf-8'><title>Explainable Clinical Classifier</title></head>"
            "<body style='background:#fafafa'>"
            "<h1 style='font-family:system-ui;margin:16px'>Explainable Clinical Note Classification</h1>"
            "<p style='font-family:system-ui;margin:0 16px;color:#555'>TF-IDF + Logistic Regression. "
            "Each prediction decomposes exactly into per-word contributions.</p>"
            + "".join(blocks) + "</body></html>")

    with open(OUT, "w") as f:
        f.write(page)
    print(f"wrote {OUT}  ({len(blocks)} examples)")


if __name__ == "__main__":
    main()

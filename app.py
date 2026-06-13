"""Interactive explainable-classifier demo (the presentation moment).

Run:  pip install streamlit
      streamlit run app.py

TF-IDF only by default. If `transformer_val_proba.csv` is present (downloaded
from the Colab ensemble run — transformer probabilities on the val notes), the
app switches to ENSEMBLE mode for the example notes: the decision + confidence
come from the blend, and the word highlights come from the interpretable
TF-IDF component (the transformer can't be word-highlighted exactly).
"""
import os

import joblib
import pandas as pd
import streamlit as st

from clinical_classifier.explain import (
    ensemble_proba,
    global_top_terms,
    highlight_html,
    token_contributions,
)

MODEL_PATH = "models/tfidf_baseline.joblib"          # held-out model (matches val)
TRANSFORMER_PROBA = "transformer_val_proba.csv"      # optional, from Colab

st.set_page_config(page_title="Clinical Note Classifier", layout="centered")
st.title("🩺 Clinical Note Classifier — explainable")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_examples():
    """label -> (id, text) from the held-out val set."""
    val = pd.read_csv("data/splits/val.csv")
    ex = {}
    for lab in ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]:
        sub = val[val.label == lab]
        if not sub.empty:
            r = sub.iloc[0]
            ex[f"Example: {lab}"] = (int(r["id"]), r["text"][:1500])
    return ex


@st.cache_data
def load_transformer_proba():
    """id -> {class: prob} from the precomputed transformer val probabilities."""
    if not os.path.exists(TRANSFORMER_PROBA):
        return None
    df = pd.read_csv(TRANSFORMER_PROBA).set_index("id")
    cols = {c: c[len("proba_"):] for c in df.columns if c.startswith("proba_")}
    return {i: {cls: float(row[c]) for c, cls in cols.items()} for i, row in df.iterrows()}


model = load_model()
examples = load_examples()
tproba = load_transformer_proba()
ensemble_on = tproba is not None

st.caption(("**Ensemble mode** — decision from TF-IDF + transformer, word evidence from "
            "the interpretable TF-IDF component." if ensemble_on else
            "TF-IDF + Logistic Regression. Drop `transformer_val_proba.csv` next to this "
            "app to enable ensemble mode."))

threshold = st.sidebar.slider("Auto-route confidence threshold", 0.3, 0.9, 0.60, 0.05)
w_tfidf = st.sidebar.slider("TF-IDF weight in ensemble", 0.0, 1.0, 0.40, 0.05,
                            disabled=not ensemble_on,
                            help="Use the best weight from ensemble.py --sweep")

src = st.radio("Note source:", ["Pick an example", "Paste your own"], horizontal=True)
if src == "Pick an example":
    choice = st.selectbox("Example note:", list(examples))
    note_id, text = examples[choice]
    st.text_area("Note:", text, height=200, disabled=True)
else:
    note_id, text = None, st.text_area("Paste a clinical note:", height=200)

if st.button("Classify", type="primary") and text and text.strip():
    use_ensemble = ensemble_on and note_id is not None and note_id in tproba
    if use_ensemble:
        _, tfidf_proba, _ = token_contributions(model, text)
        proba = ensemble_proba(tfidf_proba, tproba[note_id], w_tfidf)
        pred = next(iter(proba))
        _, _, contrib, body = highlight_html(model, text, target_class=pred)
        st.caption(f"Ensemble: {int(w_tfidf*100)}% TF-IDF + {int((1-w_tfidf)*100)}% transformer")
    else:
        pred, proba, contrib, body = highlight_html(model, text)
        if ensemble_on and src == "Paste your own":
            st.caption("Pasted note → TF-IDF only (no precomputed transformer probability).")
    conf = max(proba.values())

    if conf >= threshold:
        st.success(f"✅ **Auto-route → {pred}**  ({conf:.0%} confidence)")
    else:
        st.warning(f"⚠️ **Flag for human review** — top guess {pred}, only {conf:.0%} "
                   f"confidence (below {threshold:.0%}).")

    st.markdown("**Confidence across specialties**")
    st.bar_chart(pd.Series(proba, name="probability"))

    pos = sorted(contrib.items(), key=lambda kv: -kv[1])[:6]
    neg = sorted(contrib.items(), key=lambda kv: kv[1])[:4]
    c1, c2 = st.columns(2)
    c1.markdown("**Supports this call** 🟢\n\n" + "\n".join(f"- {w} (`{v:+.2f}`)" for w, v in pos if v > 0))
    c2.markdown("**Argues against** 🔴\n\n" + "\n".join(f"- {w} (`{v:+.2f}`)" for w, v in neg if v < 0))

    st.markdown("**The note, with TF-IDF word evidence highlighted**")
    st.markdown(body + '<p style="color:#999;font-size:12px">Green = supports, red = argues against.</p>',
                unsafe_allow_html=True)

with st.expander("What the TF-IDF model learned each specialty looks like"):
    for cls, terms in global_top_terms(model, n=12).items():
        st.markdown(f"**{cls}** — " + ", ".join(t for t, _ in terms))
